struct KalmanCache 
  prior::SecondMoment
  innovation::AbstractArray
  innovation_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

get_state(c::KalmanCache) = get_state(c.prior)
get_cov(c::KalmanCache) = get_cov(c.prior)

function KalmanCache(d::SecondMoment;m=1)
  n = dimension(d)
  innovation = zeros(m)
  innovation_cov = zeros(m,m)
  kalman_gain = zeros(n,m)
  KalmanCache(
    copy(d),
    innovation,
    innovation_cov,
    kalman_gain
    )
end

struct KalmanFilter{A<:Model,B<:Model,C<:SecondMoment} <: Filter
  transition::A 
  observation::B
  prior::C
  cache::KalmanCache
end

function KalmanFilter(transition::Model,observation::Model,prior::Distribution)
  cache = KalmanCache(prior;m=dimension(observation))
  KalmanFilter(transition,observation,prior,cache)
end

get_prior(f::KalmanFilter) = f.prior
get_transition_model(f::KalmanFilter) = f.transition
get_observation_model(f::KalmanFilter) = f.observation

function predict!(
  posterior::SecondMoment,
  f::KalmanFilter{<:StochasticAlgebraicModel},
  y::InType
  )

  x = get_state(posterior)
  P = get_cov(posterior)
  _x = get_state(f.cache)
  _P = get_cov(f.cache)

  mul!(_x,f.transition,x)
  copyto!(x,_x)

  mul!(_P,f.transition,P)
  mul!(P,_P,f.transition')
  P .+= get_cov(f.transition)

  return posterior
end

function update!(
  posterior::SecondMoment,
  f::KalmanFilter{<:Model,<:StochasticAlgebraicModel},
  y::InType
  )

  R = get_cov(f.observation)
  P = get_cov(posterior)
  x̂ = get_state(posterior)
  _P = get_cov(f.cache)

  ỹ = f.cache.innovation             
  S = f.cache.innovation_cov          
  K = f.cache.kalman_gain                       

  copyto!(ỹ,y)
  mul!(ỹ,f.observation,x̂,-1,1)             

  PHᵀ = P*f.observation'                  
  mul!(S,f.observation,PHᵀ)                    
  S .+= R                           

  F = cholesky!(S)         
  copyto!(K,PHᵀ)
  rdiv!(K,F)      

  mul!(x̂,K,ỹ,1.0,1.0)           

  mul!(_P,K,f.observation)
  _P .*= -1
  @inbounds @simd for j in axes(_P,1)
    _P[j,j] += 1
  end

  copyto!(P,_P*P*_P' + K*R*K') 

  return posterior
end
