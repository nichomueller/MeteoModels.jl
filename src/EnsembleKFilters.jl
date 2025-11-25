struct KalmanEnsemble{T,A<:AbstractMatrix{T},B<:AbstractVector{T}} <: Iterables 
  state::A
  mean::B
  anomalies::A
end

get_state(e::Ensemble) = e.state 
get_mean(e::Ensemble) = e.mean 
get_anomalies(e::Ensemble) = e.anomalies

state_size(e::KalmanEnsemble) = size(e.state,1)
ensemble_size(e::KalmanEnsemble) = size(e.state,2)

update_mean!(e::KalmanEnsemble) = mean!(e.mean,e.state)

function update_anomalies!(e::KalmanEnsemble)
  @inbounds @views for i in axes(e.anomalies)
    e.anomalies[:,i] = e.state[:,i] - e.mean
  end
end

function KalmanEnsemble(state::AbstractMatrix)
  μ = mean(state,dims=2)
  A = similar(state)
  KalmanEnsemble(state,μ,A)
end

function KalmanEnsemble(n::Int;ne=1)
  KalmanEnsemble(zeros(n,ne))
end

struct EnsembleKOperators{A<:Operators} <: Operators 
  op::A
  ensemble_size::Int
end

const EnsembleKalmanOperators{A,B,C,D,E} = EnsembleKOperators{KalmanOperators{A,B,C,D,E}}
const EnsembleUncensedKalmanOperators{A,B,C,D,E} = EnsembleKOperators{UncensedKalmanOperators{A,B,C,D,E}}

function EnsembleKOperators(args...;ensemble_size=10)
  op = KalmanOperators(args...)
  EnsembleKOperators(op,ensemble_size)
end

state_size(op::EnsembleKOperators) = state_size(op.op)
measurement_size(op::EnsembleKOperators) = measurement_size(op.op)
ensemble_size(op::EnsembleKOperators) = op.ensemble_size

function allocate_iterables(op::EnsembleKalmanOperators)
  n = state_size(op)
  ne = ensemble_size(op)
  KalmanIterables(n;ne)
end

function allocate_cache(op::EnsembleKalmanOperators)
  m = measurement_size(op)
  ne = ensemble_size(op)

  e = allocate_iterables(op)
  innovation = zeros(m,ne)
  R⁻ = cholesky(op.op.obser_noise)
  H = op.op.obser_model
  R⁻H = R⁻ * H / sqrt(ne - 1)
  ens_obs_anomalies = zeros(m,ne)
  right_etm = zeros(ne,ne)

  EnsembleKalmanCache(e,innovation,R⁻H,ens_obs_anomalies,right_etm)
end

struct EnsembleKalmanCache{K<:KalmanEnsemble,A,B,C,D} <: FilterCache
  state::K
  innovation::A
  R⁻H::B
  ens_obs_anomalies::C
  right_etm::D
end

get_state(c::EnsembleKalmanCache) = get_state(c.state)
get_mean(c::EnsembleKalmanCache) = get_mean(c) 
get_anomalies(c::EnsembleKalmanCache) = get_anomalies(c)

function predict!(
  e::KalmanEnsemble,
  c::EnsembleKalmanCache,
  op::EnsembleKalmanOperators,
  x::Observation{Controled}
  )

  @notimplemented
end

function predict!(
  e::KalmanEnsemble,
  c::EnsembleKalmanCache,
  op::EnsembleKalmanOperators,
  x::Observation
  )

  x̂ = get_state(e)
  _x̂ = get_state(c)

  mul!(_x̂,op.op.trans_model,x̂)
  copyto!(x̂,_x̂)

  update_mean!(e)
  update_anomalies!(e)
end

function update!(
  e::KalmanEnsemble,
  c::EnsembleKalmanCache,
  op::EnsembleKalmanOperators,
  x::Observation
  )

  ỹ = c.innovation 
  H = op.op.obser_model 
  x̂ = get_data(e)                     
  copyto!(ỹ,get_measurement(x))
  mul!(ỹ,H,x̂,-1,1)    
  
  A = get_anomalies(e)
  S = c.ens_obs_anomalies
  R⁻H = c.R⁻H
  mul!(S,R⁻H,A)

  K = c.right_etm
  mul!(K,S',S)
  @inbounds for i = axes(K,1)
    K[i,i] += 1
  end
  Tr = cholesky!(K)

  _A = get_anomalies(c)
  mul!(_A,Tr,A)
  copyto!(A,_A)

  μ = get_mean(e)
  _x̂ = get_state(c)
  mul!(_x̂,A,ỹ)

  @inbounds @views for i = 1:ensemble_size(e)
    x̂[:,i] = μ + _x̂[:,i]
  end
  
  return e 
end

function predict!(
  e::KalmanEnsemble,
  c::EnsembleKalmanCache,
  op::EnsembleUncensedKalmanOperators,
  x::Observation
  )

  @notimplemented
end

function update!(
  e::KalmanEnsemble,
  c::EnsembleKalmanCache,
  op::EnsembleUncensedKalmanOperators,
  x::Observation
  )

  @notimplemented
end

