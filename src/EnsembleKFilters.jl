struct KalmanEnsemble{A<:AbstractMatrix} <: Ensemble
  data::A
end

function KalmanEnsemble(n::Int;m=1)
  KalmanEnsemble(zeros(n,m))
end

get_data(e::KalmanEnsemble) = e.data
Base.copy(e::KalmanEnsemble) = KalmanEnsemble(copy(e.data))

const EnsembleKalmanOperators = EnsembleOperators{<:KalmanOperators}

function EnKFOperators(args...;ensemble_size=10)
  op = KalmanOperators(args...)
  EnsembleOperators(op,ensemble_size)
end

function allocate_iterables(op::EnsembleKalmanOperators;kwargs...)
  n = size(op.op.trans_model,1)
  KalmanEnsemble(n;kwargs...)
end

function allocate_cache(op::EnsembleKalmanOperators)
  m = measurement_size(op)
  n = state_size(op)
  ne = ensemble_size(op)
  innovation = zeros(m,ne)
  innovation_cov = diagm(ones(m))
  kalman_gain = zeros(n,m)
  mean = zeros(n)
  cov = diagm(ones(n))
  EnsembleKalmanCache(innovation,innovation_cov,kalman_gain,mean,cov)
end

struct EnsembleKalmanCache{A,B,C,D} <: EnsembleCache
  innovation::A
  innovation_cov::B
  kalman_gain::B
  mean::C 
  cov::D
end

get_mean(cache::EnsembleCache) = cache.mean 
get_cov(cache::EnsembleCache) = cache.cov 

function predict!(
  c::EnsembleCache,
  e::KalmanEnsemble,
  op::EnsembleKalmanOperators,
  x::Observation{Controled}
  )

  @notimplemented
end

# function predict!(
#   c::EnsembleCache,
#   e::KalmanEnsemble,
#   op::EnsembleKalmanOperators,
#   x::Observation
#   )

#   x̂ = get_data(e)
#   μ = get_mean(c)
#   C = get_cov(c)

#   copyto!(x̂,op.op.trans_model*x̂)

#   mean!(μ,x̂)
#   cov!(C,x̂,μ)

#   return e
# end

# function update!(
#   c::EnsembleCache,
#   e::KalmanEnsemble,
#   op::EnsembleKalmanOperators,
#   x::Observation
#   )

#   H = op.op.obser_model
#   R = op.op.obser_noise
#   x̂ = get_data(e)
#   C = get_cov(c)

#   ỹ = c.innovation             
#   S = c.innovation_cov          
#   K = c.kalman_gain                       

#   mul!(ỹ,H,x̂,-1.0,0.0)    

#   mul!(K,C,H')
#   mul!(S,H,K)
#   S .+= R                          

#   F = cholesky!(S)     
#   rdiv!(K,F)      

#   ỹ .+= get_measurement(x)
#   mul!(x̂,K,ỹ,1.0,1.0) 
  
#   return e 
# end

function predict!(
  c::EnsembleCache,
  e::KalmanEnsemble,
  op::EnsembleKalmanOperators,
  x::Observation
  )

  return e
end

function update!(
  c::EnsembleCache,
  e::KalmanEnsemble,
  op::EnsembleKalmanOperators,
  x::Observation
  )

  H = op.op.obser_model
  R = op.op.obser_noise
  x̂ = get_data(e)
  μ = get_mean(c)
  C = get_cov(c)

  ỹ = c.innovation             
  S = c.innovation_cov          
  K = c.kalman_gain                       

  mul!(ỹ,H,x̂)   
  Δy = similar(ỹ)
  anomalies!(Δy,ỹ)
  
  Δx = similar(x̂)
  anomalies!(Δx,x̂)

  Pyy = cov(Δy') + R 
  Pxy = Δx*Δy'/(size(Δy,1)-1)

  F = cholesky!(Pyy)  
  copyto!(K,Pxy)   
  rdiv!(K,F)      

  for i = axes(Δy,1), j = axes(Δy,2)
    Δy[i,j] = get_measurement(x)[i] + R[i,i]*randn() - ỹ[i,j]
  end
  mul!(Δx,K,Δy)
  
  x̂ .+= Δx
  mean!(μ,x̂)
  cov!(C,x̂,μ)
  
  return e 
end

# utils 

function anomalies!(Δ::AbstractMatrix,X::AbstractMatrix,μ::AbstractVector=mymean(X))
  @inbounds for j in axes(X,2)
    @views Δ[:, j] .= X[:, j] .- μ
  end
  return Δ
end

function cov!(C::AbstractMatrix,X::AbstractMatrix,μ::AbstractVector=mymean(X))
  δ = similar(X) 
  cov!(δ,C,X,μ) 
end

function cov!(δ::AbstractMatrix,C::AbstractMatrix,X::AbstractMatrix,μ::AbstractVector=mymean(X))
  N = size(X,2)
  @inbounds @views for i = 1:N 
    δ[:,i] = X[:,i] - μ
  end
  mul!(C,δ,δ',1/(N - 1),0.0)
  return C 
end

mymean(X::AbstractMatrix) = vec(mean(X,dims=2))