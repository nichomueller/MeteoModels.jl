struct KalmanIterables{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: Iterables
  state::A
  cov::B
end

function KalmanIterables(n::Int;state=zeros(n),cov=diagm(ones(n)))
  KalmanIterables(state,cov)
end

get_state(i::KalmanIterables) = i.state
get_cov(i::KalmanIterables) = i.cov

Base.copy(i::KalmanIterables) = KalmanIterables(copy(i.state),copy(i.cov))

struct KalmanOperators{A,B,C,D,E<:AbstractMatrix} <: Operators
  trans_model::A
  obser_model::B
  contr_model::C
  proce_noise::D
  obser_noise::E
end

function KalmanOperators(
  trans_model,
  obser_model,
  proce_noise,
  obser_noise::AbstractMatrix;
  kwargs...
  )
  
  contr_model = nothing 
  KalmanOperators(trans_model,obser_model,contr_model,proce_noise,obser_noise;kwargs...)
end

state_size(op::KalmanOperators) = size(op.obser_model,2)
measurement_size(op::KalmanOperators) = size(op.obser_model,1)

function allocate_iterables(op::KalmanOperators)
  n = state_size(op)
  KalmanIterables(n)
end

function allocate_cache(op::KalmanOperators)
  i = allocate_iterables(op)
  m = measurement_size(op)
  n = state_size(op)
  innovation = zeros(m)
  innovation_cov = zeros(m,m)
  kalman_gain = zeros(n,m)
  KalmanCache(
    i,
    innovation,
    innovation_cov,
    kalman_gain
    )
end

struct KalmanCache{T,A,B} <: FilterCache
  state::KalmanIterables{T,A,B}
  innovation::A
  innovation_cov::B
  kalman_gain::B
end

get_state(c::KalmanCache) = get_state(c.state)
get_cov(c::KalmanCache) = get_cov(c.state)

function predict!(i::KalmanIterables,cache::KalmanCache,op::KalmanOperators,x::Observation{Controled})
  x̂ = get_state(i)
  P = get_cov(i)
  _x̂ = get_state(cache)
  _P = get_cov(cache)
  control = get_control(x)

  mul!(_x̂,op.trans_model,x̂)
  mul!(x̂,op.contr_model,control)
  x̂ .+= _x̂

  mul!(_P,op.trans_model,P)
  mul!(P,_P,op.trans_model')
  P .+= op.proce_noise

  return i
end

function predict!(i::KalmanIterables,cache::KalmanCache,op::KalmanOperators,x::Observation)
  x̂ = get_state(i)
  P = get_cov(i)
  _x̂ = get_state(cache)
  _P = get_cov(cache)

  mul!(_x̂,op.trans_model,x̂)
  copyto!(x̂,_x̂)

  mul!(_P,op.trans_model,P)
  mul!(P,_P,op.trans_model')
  P .+= op.proce_noise

  return i
end

function update!(i::KalmanIterables,cache::KalmanCache,op::KalmanOperators,x::Observation)
  H = op.obser_model
  R = op.obser_noise
  P = get_cov(i)
  x̂ = get_state(i)
  _P = get_cov(cache)

  ỹ = cache.innovation             
  S = cache.innovation_cov          
  K = cache.kalman_gain                       

  copyto!(ỹ,get_measurement(x))
  mul!(ỹ,H,x̂,-1,1)             

  PHᵀ = P*H'                  
  mul!(S,H,PHᵀ)                    
  S .+= R                           

  F = cholesky!(S)         
  copyto!(K,PHᵀ)
  rdiv!(K,F)      

  mul!(x̂,K,ỹ,1.0,1.0)           

  mul!(_P,K,H)
  _P .*= -1
  @inbounds @simd for j in axes(_P,1)
    _P[j,j] += 1
  end

  copyto!(P,_P*P*_P' + K*R*K') 

  return i
end

const KalmanFilter{A<:KalmanOperators,B<:KalmanCache} = Filter{A,B}

