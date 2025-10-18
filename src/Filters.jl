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

function predict(i::KalmanIterables,op::KalmanOperators,)
  predict!(f.iterables,f.operators,args...)
  return i
end

abstract type FilterCache end

struct KalmanCache{A<:AbstractVector,B<:AbstractMatrix,C<:AbstractMatrix} <: FilterCache
  innovation::A
  innovation_cov::B
  kalman_gain::C
end

struct Filter{A<:Operators,B<:Iterables,C<:FilterCache} <: Filter
  operators::A
  iterables::B
  history::Vector{B}
  cache::C
end

get_state(f::Filter) = get_state(f.iterables)
state_size(f::Filter) = size(get_state(f))

function predict(f::Filter,args...)
  @abstractmethod
end

function update(f::Filter,args...)
  @abstractmethod
end

function evaluate(f::Filter,args...)
  @abstractmethod
end

function update_history!(f::Filter)
  push!(f.history,f.iterables)
end

const KalmanFilter{A<:KalmanOperators,B<:KalmanIterables,C<:KalmanCache} = Filter{A,B,C}

function predict(f::KalmanFilter,args...)
  update_models!(f.operators,args...)
  update_noise!(f.operators,args...)

  update!(f.cache,f.iterables,f.operators,args...)
  predict!(f.iterables,f.operators,args...)
  return f.iterables
end

function update(f::KalmanFilter,args...)
  update!(f.cache,f.iterables,f.operators,args...)
  update_history!(f)
end

function evaluate(f::KalmanFilter,args...)
  predict!(f,args...)
  update!(f,args...)
  return get_state(f)
end
