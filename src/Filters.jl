abstract type ObservationStyle end
struct Controled <: ObservationStyle end
struct UnControled <: ObservationStyle end

abstract type Observation{A<:ObservationStyle} end

get_time(o::Observation) = @abstractmethod
get_measurement(o::Observation) = @abstractmethod
get_control(o::Observation) = @abstractmethod

struct SimpleObservation{A<:Real,B<:AbstractVector} <: Observation{Controled}
  time::A 
  measurement::B
end

get_time(o::SimpleObservation) = o.time
get_measurement(o::SimpleObservation) = o.measurement

struct GenericObservation{A<:Real,B<:AbstractVector,C<:AbstractVector} <: Observation{UnControled}
  time::A 
  measurement::B
  control::C
end

get_time(o::GenericObservation) = o.time
get_measurement(o::GenericObservation) = o.measurement
get_control(o::GenericObservation) = o.control

abstract type Operators end

update!(op::Operators,args...) = @abstractmethod

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod

abstract type FilterCache end

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
  update!(f.operators,args...)
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

