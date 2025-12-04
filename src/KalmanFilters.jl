struct KalmanCache 
  obs_prior::SecondMoment
  innovation::AbstractArray
  mixed_cov::AbstractMatrix
  kalman_gain::AbstractMatrix
end

function KalmanCache(d::SecondMoment,obs_d::SecondMoment)
  n = dimension(d)
  m = dimension(obs_d)

  innovation = zeros(m)
  mixed_cov = zeros(n,m)
  kalman_gain = zeros(n,m)

  KalmanCache(copy(obs_d),innovation,mixed_cov,kalman_gain)
end

struct KalmanFilter{A<:Model,B<:Model,C<:SecondMoment} <: Filter
  transition::A 
  observation::B
  prior::C
  obs_prior::C 
  cache::KalmanCache
end

function KalmanFilter(transition::Model,observation::Model,prior::Distribution)
  obs_prior = observation(prior)
  cache = KalmanCache(prior,obs_prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

get_prior(f::KalmanFilter) = f.prior
get_observation_prior(f::KalmanFilter) = f.obs_prior
get_transition_model(f::KalmanFilter) = f.transition
get_observation_model(f::KalmanFilter) = f.observation

function kalman_gain!(f::KalmanFilter,posterior::Distribution)
  K = f.cache.kalman_gain
  obs_prior = get_observation_prior(f)
  mixed_cov!(K,posterior,obs_prior)

  Pyy = get_cov(f.cache.obs_prior) 
  copyto!(Pyy,get_cov(obs_prior))
  C = cholesky!(Pyy)
  rdiv!(K,C)

  K
end

function update!(posterior::Distribution,f::KalmanFilter,ỹ::InType)
  obs_prior = get_observation_prior(f)
  x̂ = get_state(posterior)
  Pxx = get_cov(posterior)
  Pyy = get_cov(obs_prior)
  K = f.cache.kalman_gain
  Pxy = f.cache.mixed_cov

  mul!(x̂,K,ỹ,1,1)
  mul!(Pxy,Pyy,K')
  mul!(Pxx,K,Pxy,-1,1)

  posterior
end

# function forecast!(
#   posterior::SecondMoment,
#   f::KalmanFilter{<:StochasticAlgebraicModel},
#   y::InType
#   )

#   x = get_state(posterior)
#   P = get_cov(posterior)
#   _x = get_state(f.cache)
#   _P = get_cov(f.cache)

#   mul!(_x,f.transition,x)
#   copyto!(x,_x)

#   mul!(_P,f.transition,P)
#   mul!(P,_P,f.transition')
#   P .+= get_cov(f.transition)

#   return posterior
# end

# function analyse!(
#   posterior::SecondMoment,
#   f::KalmanFilter{<:Model,<:StochasticAlgebraicModel},
#   y::InType
#   )

#   R = get_cov(f.observation)
#   P = get_cov(posterior)
#   x̂ = get_state(posterior)
#   _P = get_cov(f.cache)

#   ỹ = f.cache.innovation             
#   S = f.cache.innovation_cov          
#   K = f.cache.kalman_gain                       

#   copyto!(ỹ,y)
#   mul!(ỹ,f.observation,x̂,-1,1)             

#   PHᵀ = P*f.observation'                  
#   mul!(S,f.observation,PHᵀ)                    
#   S .+= R                           

#   F = cholesky!(S)         
#   copyto!(K,PHᵀ)
#   rdiv!(K,F)      

#   mul!(x̂,K,ỹ,1.0,1.0)           

#   mul!(_P,K,f.observation)
#   _P .*= -1
#   @inbounds @simd for j in axes(_P,1)
#     _P[j,j] += 1
#   end

#   copyto!(P,_P*P*_P' + K*R*K') 

#   return posterior
# end