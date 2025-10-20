struct KalmanIterables{T,A<:AbstractVector{T},B<:AbstractMatrix{T}} <: Iterables
  state::A
  cov::B
end

function KalmanIterables(n::Int;state=zeros(n),cov=Float64.(I(n)))
  KalmanIterables(state,cov)
end

get_state(i::KalmanIterables) = i.state
get_cov(i::KalmanIterables) = i.cov

struct KalmanCache{T,A,B,C<:Factorization} <: FilterCache
  state::KalmanIterables{T,A,B}
  innovation::A
  innovation_cov::B
  kalman_gain::B
  extra1::B 
  extra2::B 
  extra3::B 
  fact::C
end

function KalmanCache(i::KalmanIterables)
  innovation = similar(get_state(i))
  innovation_cov = similar(get_cov(i))
  kalman_gain = similar(get_cov(i))
  extra1 = similar(get_cov(i))
  extra2 = similar(get_cov(i))
  extra3 = similar(get_cov(i))
  fact = cholesky(innovation_cov'*innovation_cov)
  KalmanCache(
    i,
    innovation,
    innovation_cov,
    kalman_gain,
    extra1,
    extra2,
    extra3,
    fact
    )
end

get_state(c::KalmanCache) = get_state(c.state)
get_cov(c::KalmanCache) = get_cov(c.state)

struct KalmanOperators{T,A<:AbstractMatrix{T},B} <: Operators
  trans_model::A
  obser_model::A
  contr_model::B
  proce_noise::A
  obser_noise::A
end

function KalmanOperators(
    trans_model::AbstractMatrix,
    obser_model::AbstractMatrix,
    contr_model=nothing,
    proce_noise::AbstractMatrix=0.3*Float64.(I(size(trans_model,1))),
    obser_noise::AbstractMatrix=0.01*Float64.(I(size(trans_model,1)))
    )
    
    KalmanOperators(
      trans_model,
      obser_model,
      contr_model,
      proce_noise,
      obser_noise
    )
end

function update!(op::KalmanOperators,args...)
  return
end

function predict!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation{Controled})
  state = get_state(i)
  cov = get_cov(i)
  _state = get_state(cache)
  _cov = get_cov(cache)
  control = get_control(x)

  mul!(_state,op.trans_model,state)
  mul!(state,op.contr_model,control)
  state .+= _state

  mul!(_cov,op.trans_model,cov)
  mul!(_cov,_cov,op.trans_model')
  _cov .+= op.proce_noise

  copyto!(i.state,state)
  copyto!(i.cov,_cov)
  return i
end

function predict!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation)
  state = get_state(i)
  cov = get_cov(i)
  _state = get_state(cache)
  _cov = get_cov(cache)

  mul!(_state,op.trans_model,state)
  state .+= _state

  mul!(_cov,op.trans_model,cov)
  mul!(_cov,_cov,op.trans_model')
  _cov .+= op.proce_noise

  copyto!(i.state,state)
  copyto!(i.cov,_cov)
  return i
end

function update!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation)
  H = op.obser_model
  R = op.obser_noise
  P = get_cov(i)
  x̂ = get_state(i)

  ỹ = cache.innovation             
  S = cache.innovation_cov          
  K = cache.kalman_gain             
  A1 = cache.extra1                 
  A2 = cache.extra2
  A3 = cache.extra3
  F = cache.fact                    

  copyto!(ỹ,get_measurement(x))
  mul!(ỹ,H,x̂,-1,1)             

  mul!(A1,P,H')                   
  mul!(S,H,A1)                    
  S .+= R                           

  cholesky!(F,S)         
  copyto!(K,A1)
  rdiv!(K,F)      

  mul!(A3,K,ỹ)                   
  axpy!(1.0,A3,x̂)               

  mul!(A1,K,H)
  A1 .*= -1
  @inbounds @simd for j in axes(A1,1)
    A1[j,j] += 1
  end

  mul!(A2,A1,P)
  mul!(A3,A2,A1')
  mul!(A2,K,R)
  mul!(A1,A2,K')
  P .= A3 .+ A1                     

  return i
end

const KalmanFilter{A<:KalmanOperators,B<:KalmanIterables,C<:KalmanCache} = Filter{A,B,C}

