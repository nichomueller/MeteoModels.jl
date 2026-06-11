function execute(a::Model,prior::Law,stencil::AbstractVector)
  n = length(stencil)
  d = copy(prior)
  cache = return_cache(a,prior)
  history = Vector{typeof(d)}(undef,n)
  @inbounds for k in eachindex(stencil)
    d = evaluate!(cache,a,d)
    history[k] = copy(d)
  end
  history
end

# in differential models, the initial condition (prior) is already
# stored within the model

function execute(a::ODEModel,stencil::AbstractVector)
  p0 = a.probl.p
  u0 = a.probl.u0
  execute(a,_to_law(p0,u0),stencil)
end

function execute(a::TransientPDEModel,stencil::AbstractVector)
  p0 = a.sol.r0
  u0 = first(a.sol.us0)
  execute(a,_to_law(p0,u0),stencil)
end

function execute(a::TransientPDEModel{<:ODEParamSolution},stencil::AbstractVector)
  p0 = a.sol.r
  u0 = first(a.sol.us0)
  pspace = get_param_space(a.sol.odeop)
  constraint = ConstrainTo(pspace)
  execute(a,_to_constrained_law(p0,u0,constraint),stencil)
end

function execute(f::Filter,stencil::AbstractVector)
  model = get_transition_model(f) 
  prior = get_prior(f)
  execute(model,prior,stencil)
end

function warmup(args...)
  execute(args...)
end

function warmup!(f::Filter,stencil::AbstractVector)
  n = length(stencil)
  prior = get_prior(f)
  posterior = similar_law(prior)
  history = Vector{typeof(prior)}(undef,n)
  for _ in stencil
    forecast!(posterior,f.filter)
    copyto!(prior,posterior)
    history[k] = copy(prior)
  end
  history
end

function forecasted_history(a::Model,prior::Law,stencil::AbstractVector)
  execute(a,prior,stencil)
end

function forecasted_history(f::Filter,stencil::AbstractVector)
  execute(f,stencil)
end

function predicted_history(args...)
  loop(args...;verbose=false)
end

function forecasted_law(args...)
  h = forecasted_history(args...)
  last(h)
end

function predicted_law(args...)
  h = predicted_history(args...)
  last(h)
end

function sample_forecasted_history(args...;nsamples=1)
  h = forecasted_history(args...)
  rand(h,nsamples)
end

function sample_predicted_history(args...;nsamples=1)
  h = predicted_history(args...)
  rand(h,nsamples)
end

function sample_forecasted_law(args...)
  h = forecasted_history(args...)
  rand(h)
end

function sample_predicted_law(args...)
  h = predicted_history(args...)
  rand(h)
end

for (f,g,h) in zip(
  (:collect_forecasted_values,:collect_forecasted_value,:collect_predicted_values,:collect_predicted_value),
  (:forecasted_history,:forecasted_law,:predicted_history,:predicted_law),
  (:historical_states,:get_state,:historical_states,:get_state)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

for (f,g,h) in zip(
  (:sample_forecasted_values,:sample_forecasted_value,:sample_predicted_values,:sample_predicted_value),
  (:sample_forecasted_history,:sample_forecasted_law,:sample_predicted_history,:sample_predicted_law),
  (:stack,:identity,:stack,:identity)
  )
  @eval begin
    function $f(args...)
      $h($g(args...))
    end
  end
end

function build_linear_observation_model(
  d::Law,
  obs_ids::AbstractVector=Base.OneTo(dimension(d));
  start=1
  )

  n = length(obs_ids)
  H = zeros(n,dimension(d))
  for (j,jid) in enumerate(obs_ids)
    H[j,start+jid-1] = 1.0
  end
  Model(H)
end

# function average_forecasted_value(args...)
#   mean(collect_forecasted_values(args...))
# end

function build_true_states(args...)
  collect_forecasted_values(args...)
end

function build_true_states(μ::TransientRealisation,args...)
  μstates = repeat(vec(_get_params_marix(μ));outer=(1,num_times(μ)))
  ustates = collect_forecasted_values(args...)
  block_vcat([μstates,ustates])
end

function build_observations(f::Function,x::AbstractVector) 
  @notimplemented 
end

function build_observations(f::Function,x::AbstractMatrix;start=1) 
  @views begin
    x′ = x[start:size(x,1),:]
    y1 = f(x′[:,1])
  end
  T = eltype(y1)
  obs = zeros(T,length(y1),size(x,2))
  @inbounds @views for k in axes(obs,2)
    obs[:,k] = f(x′[:,k])
  end
  obs
end

function build_observations(f::Function,x::AbstractArray)
  build_observations(f,reshape(x,size(x,1),:))
end

function build_observations(f::Function,x::AbstractParamArray) 
  build_observations(f,get_all_data(x))
end

function build_observations(obseration::Model,obs_noise::Law,args...) 
  f(x) = observation(x) + draw(obs_noise)
  build_observations(f,args...)
end

function restart_covariance!(d::Law,P::AbstractMatrix=cov(d))
  copyto!(cov(d),P)
  d
end

# interface with stencils 

for f in (
  :execute,:warmup!,
  :forecasted_history,:forecasted_law,
  :predicted_history,:predicted_law,
  :sample_forecasted_history,:sample_forecasted_law,
  :sample_predicted_history,:sample_predicted_law,
  :collect_forecasted_values,:collect_forecasted_value,
  :collect_predicted_values,:collect_predicted_value,
  :sample_forecasted_values,:sample_forecasted_value,
  :sample_predicted_values,:sample_predicted_value
  )
  DEFAULT_PHASE = f ∈ (:warmup,:warmup!) ? WARMUP : ALL
  @eval begin
    function $f(a::Model,prior::Law,ts::TimeStencils,phase::Int=$DEFAULT_PHASE)
      x = $f(a,prior,ts[phase])
      to_stencil(x,ts,phase)
    end

    function $f(a::Model,ts::TimeStencils,phase::Int=$DEFAULT_PHASE)
      x = $f(a,ts[phase])
      to_stencil(x,ts,phase)
    end

    function $f(f::Filter,ts::TimeStencils,phase::Int=$DEFAULT_PHASE)
      x = $f(f,ts[phase])
      to_stencil(x,ts,phase)
    end
  end
end

for f in (:forecasted_history,:predicted_history)
  @eval begin
    function $f(a::StencilArray,phase::Int=a.phase)
      from_stencil(a,phase)
    end
  end
end

# utils 

function historical_states(h::AbstractVector{<:Law})
  n = length(h)
  T = typeof(get_state(h[1]))
  states = Vector{T}(undef,n)
  for k in eachindex(h)
    states[k] = get_state(h[k])
  end
  states
end

_to_law_param(p::Realisation) = Ensemble(_get_params_marix(p))
_to_law_param(p::AbstractRealisation) = _to_law_param(get_params(p))
_to_law_state(u) = FirstMoment(u)
_to_law_state(u::AbstractParamArray) = Ensemble(get_all_data(u))

_to_law(p,u) = _to_law_state(u)
_to_law(p::AbstractRealisation,u) = @notimplemented
_to_law(p::AbstractRealisation,u::AbstractParamArray) = joint_law(_to_law_param(p),_to_law_state(u))

_to_constrained_law_param(c::ConstrainTo,p::AbstractRealisation) = ConstrainedLaw(_to_law_param(p),c)
_to_constrained_law(p,u,c) = _to_law(p,u)
_to_constrained_law(p::AbstractRealisation,u::AbstractParamArray,c) = joint_law([_to_constrained_law_param(c,p),_to_law_state(u)])