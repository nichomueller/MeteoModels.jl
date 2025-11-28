struct UnscentedKalmanOperator{A,B,C,D,E<:SigmaPoints} <: Operator
  trans_model::A
  obser_model::B
  proce_noise::C
  obser_noise::D
  sigma_points::E
end

function UnscentedKalmanOperator(
  trans_model::GenericModel,
  obser_model::GenericModel,
  args...;
  kwargs...
  )
  
  n = size(trans_model,1)
  m = size(obser_model,1)
  sigma_points = SigmaPoints(2*n+m;kwargs...)
  UnscentedKalmanOperator(trans_model,obser_model,proce_noise,obser_noise,sigma_points)
end

state_size(op::UnscentedKalmanOperator) = size(op.obser_model,2)
measurement_size(op::UnscentedKalmanOperator) = size(op.obser_model,1)

function allocate_iterables(op::UnscentedKalmanOperator)
  n = state_size(op)
  KalmanIterables(n)
end

function return_cache(op::UnscentedKalmanOperator)
  i = allocate_iterables(op)
  m = measurement_size(op)
  n = state_size(op)
  innovation = zeros(m)
  innovation_cov = zeros(m,m)
  kalman_gain = zeros(n,m)
  points = zeros(n,2*n+1)
  UnscentedKalmanCache(
    i,
    innovation,
    innovation_cov,
    kalman_gain,
    points)
end

struct UnscentedKalmanCache <: FilterCache
  iter::KalmanIterables
  innovation::AbstractArray
  innovation_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
  points::AbstractVector
end

get_state(c::UnscentedKalmanCache) = get_state(c.iter)
get_cov(c::UnscentedKalmanCache) = get_cov(c.iter)

function update!(op::UnscentedKalmanOperator,i::KalmanIterables,cache::UnscentedKalmanCache,x::Observation)
  copyto!(cache.iter,i)
  update!(op.sigma_points,cache.iter)
end

function predict!(i::KalmanIterables,cache::UnscentedKalmanCache,op::UnscentedKalmanOperator,x::Observation)
  x̂ = get_state(i)
  P = get_cov(i)
  _P = get_cov(cache)
  pts = op.sigma_points
  σ = pts.points

  mul!(x̂,σ,pts.Wm)

  fill!(_P,zero(eltype(_P)))
  δ = similar(x̂)
  @views begin  
    for k in axes(σ,2)
      δ .=  x̂ - σ[:,k]
      _P[i,j] += pts.Wc[k]*δ*δ'
    end
  end
  
  axpy!(1,op.proce_noise,P)

  return i
end

function update!(i::KalmanIterables,cache::UnscentedKalmanCache,op::UnscentedKalmanOperator,x::Observation)
  R = op.obser_noise
  P = get_cov(i)
  x̂ = get_state(i)
  _x̂ = get_state(cache)
  _P = get_cov(cache)

  ỹ = cache.cache.innovation             
  S = cache.cache.innovation_cov          
  K = cache.cache.kalman_gain 

  pts = op.sigma_points
  σ = pts.points
  _σ = cache.points

  for i in eachindex(σ)
    _σ[i] = op.obser_model(σ[i])
  end
  mul!(_x̂,σ,pts.Wm)

  fill!(S,zero(eltype(S)))
  fill!(_P,zero(eltype(_P)))
  δ1 = similar(x̂)
  δ2 = similar(x̂)
  @views begin  
    for k in axes(σ,2)
      δ1 .=  x̂ - σ[:,k]
      δ2 .=  _x̂ - _σ[:,k]
      S[i,j] += pts.Wc[k]*δ2*δ2' + R[i,j]
      _P[i,j] += pts.Wc[k]*δ1*δ2'
    end
  end

  F = cholesky!(S)   
  copyto!(K,_P)
  rdiv!(K,F) 

  mul!(ỹ,K,get_measurement(x) - _x̂)
  x̂ .+= ỹ

  _P .-= K*R*K'
  copyto!(P,_P)

  return i
end

const UnscentedKalmanFilter{A<:UnscentedKalmanOperator,B<:KalmanIterables,C<:UnscentedKalmanCache} = Filter{A,B,C}

function predict!(i::Iterables,f::UnscentedKalmanFilter,obs::Observation)
  update!(i,f.operators,f.cache,obs)
  predict!(i,f.cache,f.operators,obs)
  return i
end