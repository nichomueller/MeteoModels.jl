# Head-to-head comparison of Opal'snative EchoStateNetwork (src/RC/) against
# ReservoirComputing.jl's ESN, on the same Lorenz-63 forecasting task used in
# test/ESNs.jl.
#
# Both networks are built from IDENTICAL reservoir/input weight matrices, the same
# activation, no bias, the same training window, washout, and ridge regularisation --
# so any difference in accuracy or cost reflects the software implementation, not the
# random initialisation or the hyperparameters.
#
# Part 1 reproduces the exact test/ESNs.jl setup (n = 300) and plots the closed-loop
# forecast of both networks against the true trajectory, together with the training
# and forecasting cost.
# Part 2 repeats the comparison over a range of reservoir sizes n and plots accuracy
# and cost as a function of n, which is the more informative test of the claim that
# accuracy is comparable while cost scales much better for Opal'sESN.

using Opal
using Gridap.Arrays # brings `evaluate`/`evaluate!` into scope for the closed-loop ESN call
using ReservoirComputing
using OrdinaryDiffEq
using LinearAlgebra
using Random
using Statistics
using Plots
using BenchmarkTools
using DrWatson

# leave enough room on the left for y-axis labels/ticks so they aren't chopped,
# especially in the stacked (4,1)/(3,1) layouts used below
default(left_margin=10Plots.mm)

# circle markers (Part 2) should be a solid dot, not a fill with a black outline
default(markerstrokecolor=:match)

# ------------------------------------------------------------------------
# Lorenz-63 data (identical setup to test/ESNs.jl)
# ------------------------------------------------------------------------

function lorenz!(du,u,p,t)
  du[1] = p[1]*(u[2] - u[1])
  du[2] = u[1]*(p[2] - u[3]) - u[2]
  du[3] = u[1]*u[2] - p[3]*u[3]
end

function lorenz_data(;dt=0.01,tf=200.0)
  prob = ODEProblem(lorenz!,[1.0,0.0,0.0],(0.0,tf),(10.0,28.0,8/3))
  sol = OrdinaryDiffEq.solve(prob,Tsit5();dt,saveat=dt:dt:tf)
  reduce(hcat,sol.u)
end

# ------------------------------------------------------------------------
# Build a Opal ESN and a ReservoirComputing.jl ESN that share the exact same
# reservoir matrix, input matrix, activation, and (absence of) bias. `rand_sparse` and
# `weighted_init` are ReservoirComputing.jl's own initialisers -- Opal reuses
# them internally (see src/RC/EchoStateNetworks.jl) -- so calling them once here and
# handing the resulting matrices to both constructors guarantees a byte-identical
# reservoir for both networks; only the surrounding implementation differs.
# ------------------------------------------------------------------------

function build_esn_pair(
  ninput,nstate;radius=0.9,scaling=0.1,connect=5,seed=1234,
  modifier_state=DoNotModify()
  )
  rng = MersenneTwister(seed)

  # `weighted_init` rounds `nstate` DOWN to the nearest multiple of `ninput` (each
  # input dimension gets a dedicated block of reservoir nodes); the actual size must
  # be re-derived from its output so that W_res and the reservoir/state vectors all
  # agree on the same dimension.
  W_in = weighted_init(rng,Float64,nstate,ninput;scaling)
  nstate = size(W_in,1)
  sparsity = 1.0 - connect/(nstate - 1)
  W_res = rand_sparse(rng,Float64,nstate,nstate;radius,sparsity)

  # radius/scaling are already baked into W_res/W_in, so both are set to 1 here
  # to avoid double-scaling
  esn_mm = EchoStateNetwork(
    ninput,nstate,ninput;
    radius=1.0,scaling=1.0,connect,activation=tanh,
    weights=copy(W_res),weights_in=copy(W_in),
    modifier_in=DoNotModify(),modifier_state
  )

  esn_rc = ESN(
    ninput,nstate,ninput,tanh;
    init_reservoir=(rng,dims...) -> copy(W_res),
    init_input=(rng,dims...) -> copy(W_in),
    use_bias=false
  )
  ps_rc,st_rc = setup(MersenneTwister(seed + 1),esn_rc)

  esn_mm,esn_rc,ps_rc,st_rc,nstate
end

# ------------------------------------------------------------------------
# Training / forecasting wrappers with a common signature
# ------------------------------------------------------------------------

function train_mm!(esn_mm,input_data,target_data;forget,λ)
  method = TrainRecurrentNeuralNetwork(;
    augmentation=NoAugmentation(),regularisation=NoRegularisation(),forget,λ
  )
  train(method,esn_mm,input_data,target_data)
  esn_mm
end

function train_rc(esn_rc,ps_rc,st_rc,input_data,target_data;forget,λ)
  train!(esn_rc,input_data,target_data,ps_rc,st_rc,StandardRidge(λ);forget)
end

# Opal'sclosed-loop `evaluate` echoes the seed vector back as column 1 (see
# test/ESNs.jl: `y[:,1] == test_data[:,1]`), whereas ReservoirComputing.jl's `predict`
# returns a genuinely new prediction at every column. Both are aligned below to the
# same set of forecast times, test_data columns 2:predict_len.

function forecast_mm(esn_mm,seed,steps)
  reset_state!(esn_mm)
  y = evaluate(esn_mm,seed,1:steps)
  y[:,2:end]
end

function forecast_rc(esn_rc,ps_rc,st_rc,seed,steps;rng=Random.default_rng())
  st_rc = resetcarry!(rng,esn_rc,st_rc;init_carry=zeros32)
  y, = predict(esn_rc,steps - 1,ps_rc,st_rc;initialdata=seed)
  y
end

# ------------------------------------------------------------------------
# Part 1: single-size comparison (n = 300, as in test/ESNs.jl)
# ------------------------------------------------------------------------

function part1(;nstate=300,forget=30,λ=1e-8,shift=300,train_len=5000,predict_len=1250,nsamples=5)
  data = lorenz_data()
  ninput = 3
  input_data = data[:,shift:(shift + train_len - 1)]
  target_data = data[:,(shift + 1):(shift + train_len)]
  test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]

  esn_mm,esn_rc,ps_rc,st_rc,nstate = build_esn_pair(ninput,nstate)

  train_mm!(esn_mm,input_data,target_data;forget,λ)
  ps_rc,st_rc = train_rc(esn_rc,ps_rc,st_rc,input_data,target_data;forget,λ)

  seed = test_data[:,1]
  y_mm = forecast_mm(esn_mm,seed,predict_len)
  y_rc = forecast_rc(esn_rc,ps_rc,st_rc,seed,predict_len)
  true_forecast = test_data[:,2:predict_len]

  rmse_mm = vec(sqrt.(mean(abs2,y_mm .- true_forecast,dims=1)))
  rmse_rc = vec(sqrt.(mean(abs2,y_rc .- true_forecast,dims=1)))

  labels = ("x(t)","y(t)","z(t)")
  state_plots = map(1:3) do i
    p = plot(true_forecast[i,:];label="True solution",lw=2,ylabel=labels[i])
    plot!(p,y_mm[i,:];label="Opal",ls=:dash)
    plot!(p,y_rc[i,:];label="ReservoirComputing",ls=:dot)
    p
  end
  err_plot = plot(rmse_mm;label="Opal",xlabel="Forecast step",ylabel="RMSE")
  plot!(err_plot,rmse_rc;label="ReservoirComputing")

  fig = plot(state_plots...,err_plot;layout=(4,1),size=(900,1100))
    # plot_title="Lorenz-63 closed-loop forecast (n=$nstate)")
  mkpath(datadir("plots"))
  savefig(fig,datadir("plots","compare_rc_lorenz_n$(nstate).png"))

  t_train_mm = @belapsed train_mm!($esn_mm,$input_data,$target_data;forget=$forget,λ=$λ) samples=nsamples
  t_train_rc = @belapsed train_rc($esn_rc,$ps_rc,$st_rc,$input_data,$target_data;forget=$forget,λ=$λ) samples=nsamples
  t_fcst_mm = @belapsed forecast_mm($esn_mm,$seed,$predict_len) samples=nsamples
  t_fcst_rc = @belapsed forecast_rc($esn_rc,$ps_rc,$st_rc,$seed,$predict_len) samples=nsamples

  println("n = $nstate")
  println("  training  -- Opal: $(1e3*t_train_mm) ms | RC.jl: $(1e3*t_train_rc) ms")
  println("  forecast  -- Opal: $(1e3*t_fcst_mm) ms  | RC.jl: $(1e3*t_fcst_rc) ms")
  println("  mean RMSE -- Opal: $(mean(rmse_mm))     | RC.jl: $(mean(rmse_rc))")

  (;rmse_mm,rmse_rc,t_train_mm,t_train_rc,t_fcst_mm,t_fcst_rc)
end

# ------------------------------------------------------------------------
# Part 2: sweep over reservoir size n
# ------------------------------------------------------------------------

function part2(;
  ns=(50,100,200,400,800,1600),
  forget=30,λ=1e-8,shift=300,train_len=5000,predict_len=1250,nsamples=3
  )

  data = lorenz_data()
  ninput = 3
  input_data = data[:,shift:(shift + train_len - 1)]
  target_data = data[:,(shift + 1):(shift + train_len)]
  test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]
  seed = test_data[:,1]
  true_forecast = test_data[:,2:predict_len]

  n_ns = length(ns)
  ns_actual = zeros(Int,n_ns)
  rmse_mm,rmse_rc = zeros(n_ns),zeros(n_ns)
  t_train_mm,t_train_rc = zeros(n_ns),zeros(n_ns)
  t_fcst_mm,t_fcst_rc = zeros(n_ns),zeros(n_ns)
  m_train_mm,m_train_rc = zeros(n_ns),zeros(n_ns)
  m_fcst_mm,m_fcst_rc = zeros(n_ns),zeros(n_ns)

  for (i,n) in enumerate(ns)
    esn_mm,esn_rc,ps_rc,st_rc,n_actual = build_esn_pair(ninput,n)
    ns_actual[i] = n_actual

    train_mm!(esn_mm,input_data,target_data;forget,λ)
    ps_rc,st_rc = train_rc(esn_rc,ps_rc,st_rc,input_data,target_data;forget,λ)

    y_mm = forecast_mm(esn_mm,seed,predict_len)
    y_rc = forecast_rc(esn_rc,ps_rc,st_rc,seed,predict_len)

    rmse_mm[i] = sqrt(mean(abs2,y_mm .- true_forecast))
    rmse_rc[i] = sqrt(mean(abs2,y_rc .- true_forecast))

    # @benchmark (rather than @belapsed) so both time and memory are read off
    # the same trial instead of re-running the benchmark twice
    trial_train_mm = @benchmark train_mm!($esn_mm,$input_data,$target_data;forget=$forget,λ=$λ) samples=nsamples
    trial_train_rc = @benchmark train_rc($esn_rc,$ps_rc,$st_rc,$input_data,$target_data;forget=$forget,λ=$λ) samples=nsamples
    trial_fcst_mm = @benchmark forecast_mm($esn_mm,$seed,$predict_len) samples=nsamples
    trial_fcst_rc = @benchmark forecast_rc($esn_rc,$ps_rc,$st_rc,$seed,$predict_len) samples=nsamples

    t_train_mm[i] = minimum(trial_train_mm).time / 1e9
    t_train_rc[i] = minimum(trial_train_rc).time / 1e9
    t_fcst_mm[i] = minimum(trial_fcst_mm).time / 1e9
    t_fcst_rc[i] = minimum(trial_fcst_rc).time / 1e9

    m_train_mm[i] = trial_train_mm.memory / 1e6
    m_train_rc[i] = trial_train_rc.memory / 1e6
    m_fcst_mm[i] = trial_fcst_mm.memory / 1e6
    m_fcst_rc[i] = trial_fcst_rc.memory / 1e6

    println("n=$n_actual done -- RMSE mm=$(rmse_mm[i]) rc=$(rmse_rc[i]) | train mm=$(t_train_mm[i])s rc=$(t_train_rc[i])s")
  end

  ns_v = ns_actual

  p_time = plot(ns_v,t_train_mm;ylabel="Time [s]",
    label="Opal (training)",marker=:circle,xscale=:log10,yscale=:log10,color=1)
  plot!(p_time,ns_v,t_fcst_mm;label="Opal (forecast)",marker=:circle,ls=:dash,color=1)
  plot!(p_time,ns_v,t_train_rc;label="ReservoirComputing (training)",marker=:circle,color=2)
  plot!(p_time,ns_v,t_fcst_rc;label="ReservoirComputing (forecast)",marker=:circle,ls=:dash,color=2)

  p_mem = plot(ns_v,m_train_mm;xlabel="Reservoir size",ylabel="Memory [Mb]",
    label="Opal (training)",marker=:circle,xscale=:log10,yscale=:log10,color=1)
  plot!(p_mem,ns_v,m_fcst_mm;label="Opal (forecast)",marker=:circle,ls=:dash,color=1)
  plot!(p_mem,ns_v,m_train_rc;label="ReservoirComputing (training)",marker=:circle,color=2)
  plot!(p_mem,ns_v,m_fcst_rc;label="ReservoirComputing (forecast)",marker=:circle,ls=:dash,color=2)

  fig = plot(p_time,p_mem;layout=(2,1),size=(800,900))
  mkpath(datadir("plots"))
  savefig(fig,datadir("plots","compare_rc_sweep.png"))

  (;ns=ns_actual,rmse_mm,rmse_rc,
    t_train_mm,t_train_rc,t_fcst_mm,t_fcst_rc,
    m_train_mm,m_train_rc,m_fcst_mm,m_fcst_rc)
end

# ------------------------------------------------------------------------
# Part 3: does RecycleValidation-tuned training improve Opal'sESN accuracy
# relative to ReservoirComputing.jl's (untuned) ESN?
#
# RecycleValidation performs a cross-validated grid search (refined by a short
# Nelder-Mead local search) over the ESN's spectral radius and input scaling --
# and, optionally, the ridge Tikhonov parameter -- using held-out windows carved
# out of the training data itself, so no separate validation set is needed and
# the reservoir/input matrices are never touched. ReservoirComputing.jl has no
# built-in equivalent, so it is trained exactly as in Part 1/2 for reference; only
# Opal'sown ESN is retrained here, with and without RecycleValidation.
# ------------------------------------------------------------------------

function train_mm_rv!(
  esn_mm,input_data,target_data;forget,λ,
  radius_range=range(0.5,1.05,length=60),
  scaling_range=range(0.05,3.0,length=60),
  λ_range=nothing,
  Nfolds=30,Nvalidation=33
  )

  method = TrainRecurrentNeuralNetwork(;
    augmentation=NoAugmentation(),regularisation=NoRegularisation(),forget,λ
  )
  Ntrain = size(input_data,2)
  rv_method = if λ_range === nothing
    RecycleValidation(method,radius_range,scaling_range;Nfolds,Ntrain,Nvalidation)
  else
    RecycleValidation(method,λ_range,radius_range,scaling_range;Nfolds,Ntrain,Nvalidation)
  end
  train(rv_method,esn_mm,input_data,target_data)
  esn_mm
end

function part3(;
  nstate=300,forget=30,λ=1e-8,shift=300,train_len=5000,predict_len=1250,
  radius_range=range(0.5,1.05,length=60),
  scaling_range=range(0.05,3.0,length=60),
  λ_range=nothing,Nfolds=30,Nvalidation=33
  )

  data = lorenz_data()
  ninput = 3
  input_data = data[:,shift:(shift + train_len - 1)]
  target_data = data[:,(shift + 1):(shift + train_len)]
  test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]
  seed = test_data[:,1]
  true_forecast = test_data[:,2:predict_len]

  # (a) Opal ESN, plain fixed-hyperparameter training, and (c) the
  # ReservoirComputing.jl reference -- exactly as built/trained in Part 1/2
  esn_mm_plain,esn_rc,ps_rc,st_rc,nstate = build_esn_pair(ninput,nstate)
  train_mm!(esn_mm_plain,input_data,target_data;forget,λ)
  ps_rc,st_rc = train_rc(esn_rc,ps_rc,st_rc,input_data,target_data;forget,λ)

  # (b) Opal ESN, RecycleValidation-tuned training. A fresh, *unscaled*
  # reservoir (radius=scaling=1 baked into the matrices) is used so that the
  # tuned esn_mm_rv.radius[]/scaling[] found below are directly the effective
  # spectral radius / input scaling, not a multiplier on top of Part 1's 0.9/0.1.
  esn_mm_rv, = build_esn_pair(ninput,nstate;radius=1.0,scaling=1.0,seed=4321)
  train_mm_rv!(
    esn_mm_rv,input_data,target_data;
    forget,λ,radius_range,scaling_range,λ_range,Nfolds,Nvalidation
  )

  y_mm_plain = forecast_mm(esn_mm_plain,seed,predict_len)
  y_mm_rv = forecast_mm(esn_mm_rv,seed,predict_len)
  y_rc = forecast_rc(esn_rc,ps_rc,st_rc,seed,predict_len)

  rmse_mm_plain = vec(sqrt.(mean(abs2,y_mm_plain .- true_forecast,dims=1)))
  rmse_mm_rv = vec(sqrt.(mean(abs2,y_mm_rv .- true_forecast,dims=1)))
  rmse_rc = vec(sqrt.(mean(abs2,y_rc .- true_forecast,dims=1)))

  labels = ("x(t)","y(t)","z(t)")
  state_plots = map(1:3) do i
    p = plot(true_forecast[i,:];label="True solution",lw=2,ylabel=labels[i])
    plot!(p,y_mm_plain[i,:];label="Opal (plain)",ls=:dash)
    plot!(p,y_mm_rv[i,:];label="Opal (RV)",ls=:dashdot)
    plot!(p,y_rc[i,:];label="ReservoirComputing",ls=:dot)
    p
  end
  err_plot = plot(rmse_mm_plain;label="Opal (plain)",xlabel="Forecast step",ylabel="RMSE")
  plot!(err_plot,rmse_mm_rv;label="Opal (RV)")
  plot!(err_plot,rmse_rc;label="ReservoirComputing")

  fig = plot(state_plots...,err_plot;layout=(4,1),size=(900,1100))
    # plot_title="Effect of RecycleValidation on Opal'sESN (n=$nstate)")
  mkpath(datadir("plots"))
  savefig(fig,datadir("plots","compare_rc_recyclevalidation_n$(nstate).png"))

  println("n = $nstate")
  println("  mean RMSE -- Opal (plain):             $(mean(rmse_mm_plain))")
  println("  mean RMSE -- Opal (RV):  $(mean(rmse_mm_rv))")
  println("  mean RMSE -- ReservoirComputing:            $(mean(rmse_rc))")
  println("  tuned radius = $(esn_mm_rv.radius[]), tuned scaling = $(esn_mm_rv.scaling[])")

  (;rmse_mm_plain,rmse_mm_rv,rmse_rc,
    tuned_radius=esn_mm_rv.radius[],tuned_scaling=esn_mm_rv.scaling[])
end

# ------------------------------------------------------------------------
# Run all three parts. Part 1 mirrors test/ESNs.jl exactly (n=300, 5000-step
# training window, 1250-step forecast). Part 2 sweeps n and takes noticeably
# longer since a fresh pair of networks is trained and benchmarked at every
# size. Part 3 additionally trains Opal'sESN with RecycleValidation,
# which is itself considerably more expensive than plain training (it is a
# hyperparameter search, not a single ridge-regression solve) -- this cost is
# expected and is not part of the Part 1/2 cost comparison.
# ------------------------------------------------------------------------

results1 = part1()
results2 = part2()
results3 = part3()
