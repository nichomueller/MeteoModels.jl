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

# utils 

_to_ref(a) = a 
_to_ref(a::Number) = Ref(a)

_from_ref(a) = a 
_from_ref(a::Base.RefValue) = a[]

_set_ref!(a,b) = copyto!(a,b)
_set_ref!(a::Base.RefValue,b::Number) = (a[] = b)