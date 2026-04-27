using MeteoModels
using OrdinaryDiffEq
using Test
using LinearAlgebra
using Gridap
using Gridap.Arrays
using Random

n = 3
x = rand(n)
a = 2
factor = rand(n)
transformation = T₂()
normalisation = Normalisation(factor)
bias = AddBias(a)
modifier = Modifier(normalisation,transformation,bias)

function lorenz!(du,u,p,t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

dt = 0.01
prob = ODEProblem(lorenz!,[1.0,0.0,0.0],(0.0,200.0),(10.0,28.0,8/3))
data = OrdinaryDiffEq.solve(prob,Tsit5();dt,saveat=dt:dt:200.0)
data = reduce(hcat,data.u)

shift = 300
train_len = 5000
predict_len = 1250

input_data = data[:,shift:(shift + train_len - 1)]
target_data = data[:,(shift + 1):(shift + train_len)]
test_data = data[:,(shift + train_len + 1):(shift + train_len + predict_len)]

washout = 100
uwash = input_data[:,1:washout]
utrain = input_data[:,(washout + 1):end]
ytrain = target_data[:,(washout + 1):end] 

nstate = 300
ninput = 3
radius = 1.0
connect = 6
sparsity= 1.0-connect/(nstate-1)
scaling = 0.1

esn = EchoStateNetwork(
    ninput,nstate,ninput;
    radius,
    sparsity,
    scaling,
    modifier_in=Modifier(bias=AddBias(0.1),normalisation=Normalisation(fill(1.0,ninput))),
    modifier_state=Modifier(bias=AddBias(1.0)),
    activation=tanh
)

# novoa's esn 

novoa_esn = NovoaESN(ninput,nstate;connect,rho=radius,sigma_in=scaling,
    bias_in= [0.1],bias_out = [1.0],norm_vec = ones(ninput)
)

# train(NovoaTrainMethod(),novoa_esn,input_data,target_data)

copyto!(esn.weights, novoa_esn.W)
copyto!(esn.weights_in, novoa_esn.Win)

method = TrainRecurrentNeuralNetwork(
    augmentation=NoAugmentation(),
    regularisation=NoRegularisation(),
    washout=washout,λ=1e-16
)

# s = train(method,esn,input_data,target_data)

# rvmethod = RecycleValidation(method,radius_ranges,scaling_ranges;Nfolds,Ntrain,Nvalidation)
# train(rvmethod,esn,input_data,target_data)

radius_ranges = range(0.7,1.05,length=4)
scaling_ranges = range(LogNumber{10}(log10(1e-5)),LogNumber{10}(log10(1.0)),length=4)
tikhonov = [1e-16,1e-12,1e-10,1e-8]
rcv = RecycleValidation(
    method,tikhonov,radius_ranges,scaling_ranges;
    Nfolds=4,Ntrain=size(utrain,2),Nvalidation=200
)
p = first(rcv.updates)
a = esn
MeteoModels.replace_rv_parameters!(a,p)
# loss,λ = _rv_train!(cache,rcv,a,x,y)
c1,c2,c3,c4,c5,c6,c7 = MeteoModels.train_cache(rcv,a,utrain,ytrain)
t = rcv.method

x,y = utrain,ytrain
x′ = evaluate!(c1,t.augmentation,x)
y′ = evaluate!(c2,t.augmentation,y)

MeteoModels.reset_state!(a) 
s′ = evaluate!(c3,TrainableNetwork(a),x′)

xwash = apply_washout(x′,t.washout) 
swash = apply_washout(s′,t.washout) 
ywash = apply_washout(y′,t.washout)
ywash′ = evaluate!(c4,t.regularisation,ywash)

W, = get_parameters(a)
_fill_gram!(c5,swash,ywash′)

λvec = rcv.updates.tikhonov

best_W, = c7
local best_λ
best_loss = Inf
#   for λ in λvec
Algebra.solve!(W,RidgeRegression(λ),c5)

# novoa_s = train(NovoaTrainMethod(),novoa_esn,uwash,utrain,ytrain)

U_wash,U_train,Y_train = uwash,utrain,ytrain

U_wash_arr  = MeteoModels._ensure3d(U_wash)
U_train_arr = MeteoModels._ensure3d(U_train)
Y_train_arr = MeteoModels._ensure3d(Y_train)

N_dim, _, N_traj = size(U_train_arr)
@assert N_dim == novoa_esn.N_dim        "U_train first dim $N_dim ≠ ESN N_dim $(a.N_dim)"
@assert size(Y_train_arr) == size(U_train_arr) "U_train and Y_train must have the same shape"

# ── Normalization from training data across all trajectories ──────────────
lo = fill( Inf, N_dim)
hi = fill(-Inf, N_dim)
for kk in 1:N_traj
    lo = min.(lo, vec(minimum(view(U_train_arr, :, :, kk), dims=2)))
    hi = max.(hi, vec(maximum(view(U_train_arr, :, :, kk), dims=2)))
end
nrm = hi .- lo
nrm[nrm .== 0.0] .= 1.0
novoa_esn.norm_vec .= nrm

# ── Coarse grid search ────────────────────────────────────────────────────
m = NovoaTrainMethod()
rho_vals = range(m.rho_range[1],       m.rho_range[2];       length=m.N_grid)
sin_vals = range(m.sigin_log_range[1], m.sigin_log_range[2]; length=m.N_grid)

best_mse  = Inf
best_rho  = Float64(rho_vals[1])
best_sin  = Float64(sin_vals[1])
best_tikh = m.tikh_values[1]

rho = rho_vals[1]
log10_sin = sin_vals[1]
# mse, tikh, _ = MeteoModels._rvc_noise(
#         Float64(rho), Float64(log10_sin), m.tikh_values,
#         U_wash_arr, U_train_arr, Y_train_arr,
#         novoa_esn.norm_vec, novoa_esn.W, novoa_esn.Win, novoa_esn.bias_in, novoa_esn.bias_out,
#         m.N_folds, m.N_val, m.noise_level,
#     )

tikh_values = m.tikh_values
norm_vec = novoa_esn.norm_vec
W, Win = novoa_esn.W, novoa_esn.Win
bias_in, bias_out = novoa_esn.bias_in, novoa_esn.bias_out
N_folds, N_val, noise_level = m.N_folds, m.N_val, m.noise_level
sigma_in               = 10.0^log10_sin
N_dim, N_train, N_traj = size(U_train_arr)
N_units                = size(W, 1)
N_aug                  = N_units + length(bias_out)
n_tikh                 = length(tikh_values)
norm_sq                = norm_vec .^ 2

# ── Open-loop drive for every trajectory; accumulate LHS, RHS ────────────
Xa_all = Vector{Matrix{Float64}}(undef, N_traj)
LHS    = zeros(N_aug, N_aug)
RHS    = zeros(N_aug, N_dim)

kk = 1
Xa_w  = MeteoModels._nova_open_loop(zeros(N_units), view(U_wash_arr, :, :, kk),
                            W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
r_end = Xa_w[1:N_units, end]

# Training open-loop starting from washout end state
Xa          = MeteoModels._nova_open_loop(r_end, view(U_train_arr, :, :, kk),
                                W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
Xa_all[kk]  = Xa    # (N_aug × N_train+1)

# Add noise to targets (RVC-Noise trick); use per-trajectory seed
Yt    = copy(view(Y_train_arr, :, :, kk))   # (N_dim × N_train)
U_std = vec(std(Yt, dims=2))
rng   = MersenneTwister(kk)
for i in 1:N_dim
    Yt[i, :] .+= randn(rng, N_train) .* (noise_level * U_std[i])
end

# Xa[:, 2:end]: states driven by training inputs (columns 2..N_train+1)
X_reg = Xa[:, 2:end]
LHS  .+= X_reg * X_reg'
RHS  .+= X_reg * Yt'

# ── Solve for all tikh values (incremental diagonal update) ───────────────
LHS_work = copy(LHS)
Wout_all = Vector{Matrix{Float64}}(undef, n_tikh)
for j in 1:n_tikh
    δ = j == 1 ? tikh_values[1] : tikh_values[j] - tikh_values[j-1]
    @views LHS_work[diagind(LHS_work)] .+= δ
    Wout_all[j] = Matrix(LHS_work \ RHS)   # (N_aug × N_dim)
end


for rho in rho_vals, log10_sin in sin_vals
    mse, tikh, _ = MeteoModels._rvc_noise(
        Float64(rho), Float64(log10_sin), m.tikh_values,
        U_wash_arr, U_train_arr, Y_train_arr,
        novoa_esn.norm_vec, novoa_esn.W, novoa_esn.Win, novoa_esn.bias_in, novoa_esn.bias_out,
        m.N_folds, m.N_val, m.noise_level,
    )
    if mse < best_mse
        best_mse  = mse
        best_rho  = Float64(rho)
        best_sin  = Float64(log10_sin)
        best_tikh = tikh
    end
end

# ── Nelder-Mead local refinement ─────────────────────────────────────────
optfun = x -> begin
    mse, _, _ = MeteoModels._rvc_noise(
        x[1], x[2], m.tikh_values,
        U_wash_arr, U_train_arr, Y_train_arr,
        novoa_esn.norm_vec, novoa_esn.W, novoa_esn.Win, novoa_esn.bias_in, novoa_esn.bias_out,
        m.N_folds, m.N_val, m.noise_level,
    )
    mse
end
result = optimize(optfun, [best_rho, best_sin], NelderMead(),
                    Optim.Options(iterations=8, show_trace=false))
if Optim.minimum(result) < best_mse
    best_rho = Optim.minimizer(result)[1]
    best_sin = Optim.minimizer(result)[2]
    best_mse, best_tikh, _ = MeteoModels._rvc_noise(
        best_rho, best_sin, m.tikh_values,
        U_wash_arr, U_train_arr, Y_train_arr,
        novoa_esn.norm_vec, novoa_esn.W, novoa_esn.Win, novoa_esn.bias_in, novoa_esn.bias_out,
        m.N_folds, m.N_val, m.noise_level,
    )
end