""" 
    abstract type ObservationStyle end

Subtypes:
- [`Controled`](@ref)
- [`UnControled`](@ref)
"""
abstract type ObservationStyle end

""" 
    struct Controled <: ObservationStyle end

Trait of observations 
"""
struct Controled <: ObservationStyle end

""" 
    struct UnControled <: ObservationStyle end
"""
struct UnControled <: ObservationStyle end

abstract type Observation{A<:ObservationStyle} end

get_time(o::Observation) = @abstractmethod
get_measurement(o::Observation) = @abstractmethod
get_control(o::Observation) = @abstractmethod

Observation(args...) = @abstractmethod

struct GenericObservation{A,B} <: Observation{UnControled}
  time::A 
  measurement::B
end

function Observation(time,measurement)
  GenericObservation(_to_ref(time),_to_ref(measurement))
end

get_time(o::GenericObservation) = _from_ref(o.time)
get_measurement(o::GenericObservation) = _from_ref(o.measurement)

function update!(o::GenericObservation,t,z)
  _set_ref!(o.time,t)
  _set_ref!(o.measurement,z)
end

struct ControledObservation{A,B,C} <: Observation{Controled}
  time::A
  measurement::B
  control::C
end

function Observation(time,measurement,control)
  ControledObservation(_to_ref(time),_to_ref(measurement),_to_ref(control))
end

get_time(o::ControledObservation) = _from_ref(o.time)
get_measurement(o::ControledObservation) = _from_ref(o.measurement)
get_control(o::ControledObservation) = _from_ref(o.control)

function update!(o::GenericObservation,t,z,u)
  _set_ref!(o.time,t)
  _set_ref!(o.measurement,z)
  _set_ref!(o.control,u)
end

abstract type Operators end

state_size(op::Operators) = @abstractmethod
measurement_size(op::Operators) = @abstractmethod

update!(op::Operators,args...) = @abstractmethod

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod

abstract type FilterCache end

allocate_iterables(op::Operators) = @abstractmethod
allocate_cache(op::Operators) = @abstractmethod

struct Filter{A<:Operators,B<:Iterables,C<:FilterCache} 
  operators::A
  iterables::B
  history::Vector{B}
  cache::C
end

function Filter(op::Operators,i::A) where A<:Iterables
  history = A[]
  cache = allocate_cache(op)
  Filter(op,i,history,cache)
end

(f::Filter)(args...) = evaluate(f,args...)

get_state(f::Filter) = get_state(f.iterables)
state_size(f::Filter) = size(get_state(f))

function predict!(f::Filter,args...)
  update!(f.operators,args...)
  predict!(f.cache,f.iterables,f.operators,args...)
  return f.iterables
end

function update!(f::Filter,args...)
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

# utils 

_to_ref(a) = a 
_to_ref(a::Number) = Ref(a)

_from_ref(a) = a 
_from_ref(a::Base.RefValue) = a[]

_set_ref!(a,b) = copyto!(a,b)
_set_ref!(a::Base.RefValue,b::Number) = (a[] = b)