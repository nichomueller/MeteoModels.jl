abstract type Operators end

update_models!(op::Operators,args...) = @abstractmethod
update_noise!(op::Operators,args...) = @abstractmethod

struct KalmanOperators{A,B,C,D,E} <: Operators
  trans_model::A
  contr_model::B
  obser_model::C
  proce_noise::D
  obser_noise::E
end

function update_models!(op::KalmanOperators,args...)
  update!(op.trans_model,args...)
  update!(op.contr_model,args...)
  update!(op.obser_model,args...)
end

function update_noise!(op::KalmanOperators,args...)
  update!(op.proce_noise,args...)
  update!(op.obser_noise,args...)
end

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod

struct KalmanIterables{A<:AbstractVector,B<:AbstractMatrix} <: Iterables
  state::A
  cov::B
end

get_state(i::KalmanIterables) = i.state
get_cov(i::KalmanIterables) = i.cov

abstract type FilterCache end

struct KalmanCache{A<:Iterables,B<:AbstractVector,C<:AbstractMatrix,D<:Factorization} <: FilterCache
  state::A
  innovation::B
  innovation_cov::C
  kalman_gain::C
  extra1::C 
  extra2::C 
  extra3::C 
  fact::D
end

get_state(c::KalmanCache) = get_state(c.state)
get_cov(c::KalmanCache) = get_cov(c.state)

predict!(args...) = update!(args...)

function predict!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation)
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

function predict!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation{Controled})
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
  ldiv!(A2,F,A1')                
  ldiv!(K,F',A2)                 

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
  mul!(A2,A2,K')
  P .= A3 .+ A2                     

  return i
end

struct Filter{A<:Operators,B<:Iterables,C<:FilterCache} 
  operators::A
  iterables::B
  history::Vector{B}
  cache::C
end

(f::Filter)(args...) = evaluate(f,args...)

get_state(f::Filter) = get_state(f.iterables)
state_size(f::Filter) = size(get_state(f))

function predict(f::Filter,args...)
  update_models!(f.operators,args...)
  update_noise!(f.operators,args...)
  predict!(f.cache,f.iterables,f.operators,args...)
  return f.iterables
end

function update(f::Filter,args...)
  update!(f.cache,f.iterables,f.operators,args...)
  update_history!(f)
end

function evaluate(f::Filter,args...)
  predict!(f,args...)
  update!(f,args...)
  return get_state(f)
end

function update_history!(f::Filter)
  push!(f.history,deepcopy(f.iterables))
end

const KalmanFilter{A<:KalmanOperators,B<:KalmanIterables,C<:KalmanCache} = Filter{A,B,C}

