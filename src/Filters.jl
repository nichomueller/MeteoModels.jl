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
  mul!(_cov,cov,op.trans_model')
  mul!(cov,op.trans_model,_cov)
  cov .+= op.obser_noise

  copyto!(i.state,state)
  copyto!(i.cov,cov)
  return i
end

function predict!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation{Controled})
  state = get_state(i)
  cov = get_cov(i)
  _state = get_state(cache)
  _cov = get_cov(cache)

  mul!(_state,op.trans_model,state)
  mul!(_cov,cov,op.trans_model')
  mul!(cov,op.trans_model,_cov)
  cov .+= op.proce_noise

  copyto!(i.state,_state)
  copyto!(i.cov,cov)
  return i
end

function update!(cache::KalmanCache,i::KalmanIterables,op::KalmanOperators,x::Observation)
  _state = get_state(cache)
  _cov = get_cov(cache)
  meas = get_measurement(x)

  copyto!(cache.innovation,meas)
  mul!(cache.innovation,op.obser_model,i.state,-1,0)
  mul!(cache.extra1,i.cov,op.obser_model')
  mul!(cache.innovation_cov,op.obser_model,cache.extra1)
  axpy!(1,cache.innovation_cov,op.obser_noise)

  cholesky!(cache.fact,cache.innovation_cov)
  mul!(cache.extra1,i.cov,op.obser_model)
  ldiv!(cache.extra2,cache.fact',cache.extra1')
  transpose!(cache.kalman_gain,cache.extra2)

  mul!(_state,cache.kalman_gain,cache.innovation)
  axpy!(1,i.state,_state)

  mul!(cache.extra1,cache.kalman_gain,op.obser_model)
  cache.extra1 .+= I 
  mul!(cache.extra2,cache.kalman_gain,op.obser_noise)
  mul!(cache.extra3,cache.extra2,op.kalman_gain')
  mul!(cache.extra2,cache.extra1,i.cov)
  mul!(_cov,cache.extra2,cache.extra1')
  i.cov .= _cov .+ cache.extra3
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
  push!(f.history,f.iterables)
end

const KalmanFilter{A<:KalmanOperators,B<:KalmanIterables,C<:KalmanCache} = Filter{A,B,C}

