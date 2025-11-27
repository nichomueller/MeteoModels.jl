struct KalmanEnsemble{T,A<:AbstractMatrix{T},B<:AbstractVector{T}} <: Iterables 
  state::A
  mean::B
  anomalies::A
end

get_state(e::KalmanEnsemble) = e.state 
get_mean(e::KalmanEnsemble) = e.mean 
get_anomalies(e::KalmanEnsemble) = e.anomalies

state_size(e::KalmanEnsemble) = size(e.state,1)
ensemble_size(e::KalmanEnsemble) = size(e.state,2)

Base.copy(e::KalmanEnsemble) = KalmanEnsemble(copy(e.state),copy(e.mean),copy(e.anomalies))

update_mean!(e::KalmanEnsemble) = mean!(e.mean,e.state)

function update_anomalies!(e::KalmanEnsemble)
  @inbounds @views for i in axes(e.anomalies,2)
    e.anomalies[:,i] = e.state[:,i] - e.mean
  end
end

function update_state!(e::KalmanEnsemble)
  @inbounds @views for i in axes(e.anomalies,2)
    e.state[:,i] = e.anomalies[:,i] + e.mean
  end
end

function KalmanEnsemble(state::AbstractMatrix)
  n = size(state,1)
  μ = similar(state,(n,))
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
const EnsembleUncensedKalmanOperators{A,B,C,D,E} = EnsembleKOperators{UnscentedKalmanOperators{A,B,C,D,E}}

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
  KalmanEnsemble(n;ne)
end

function allocate_cache(op::EnsembleKalmanOperators)
  n = state_size(op)
  m = measurement_size(op)
  ne = ensemble_size(op)

  e = allocate_iterables(op)
  innovation = zeros(m)
  innovation_cov = zeros(m,m)
  R⁻ = cholesky(op.op.obser_noise)
  H = op.op.obser_model
  R⁻H = (R⁻.U \ H) / sqrt(ne - 1)
  AHᵀ = zeros(ne,m)
  PHᵀ = zeros(n,m)
  kalman_gain = zeros(n,m)
  ens_obs_anomalies = zeros(m,ne)
  right_etm = zeros(ne,ne)

  EnsembleKalmanCache(
    e,
    innovation,
    innovation_cov,
    AHᵀ,PHᵀ,
    kalman_gain,
    R⁻H,
    ens_obs_anomalies,
    right_etm
  )
end

struct EnsembleKalmanCache{K<:KalmanEnsemble,A,B,C,D,E,F,G,H} <: FilterCache
  state::K
  innovation::A
  innovation_cov::B
  AHᵀ::C
  PHᵀ::D
  kalman_gain::E
  R⁻H::F
  ens_obs_anomalies::G
  right_etm::H
end

get_state(c::EnsembleKalmanCache) = get_state(c.state)
get_mean(c::EnsembleKalmanCache) = get_mean(c.state) 
get_anomalies(c::EnsembleKalmanCache) = get_anomalies(c.state)

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

  # step 1: compute and apply kalman gain on ensemble average 
  ne = ensemble_size(e)
  ỹ = c.innovation 
  H = op.op.obser_model 
  μ = get_mean(e)
  A = get_anomalies(e)
  AHᵀ = c.AHᵀ
  PHᵀ = c.PHᵀ
  S = c.innovation_cov
  K = c.kalman_gain
  R = op.op.obser_noise

  mul!(AHᵀ,A',H')
  mul!(PHᵀ,A,AHᵀ,1 / (ne - 1),0.0)
  mul!(S,H,PHᵀ)                    
  S .+= R                           

  F = cholesky!(S)         
  copyto!(K,PHᵀ)
  rdiv!(K,F)

  copyto!(ỹ,get_measurement(x))
  mul!(ỹ,H,μ,-1,1) 
  mul!(μ,K,ỹ,1,1) 
  
  # step 2: update right ensemble transform matrix 
  HA = c.ens_obs_anomalies
  R⁻H = c.R⁻H
  mul!(HA,R⁻H,A)

  _Tr = c.right_etm
  mul!(_Tr,HA',HA)
  @inbounds for i = axes(_Tr,1)
    _Tr[i,i] += 1
  end
  Tr = cholesky!(_Tr)
  rdiv!(A,Tr.U)

  # step 3: offset anomalies to retrieve ensemble 
  update_state!(e)
  
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

