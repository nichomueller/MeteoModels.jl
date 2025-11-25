abstract type Operators end

state_size(op::Operators) = @abstractmethod
measurement_size(op::Operators) = @abstractmethod

allocate_iterables(op::Operators) = @abstractmethod
allocate_cache(op::Operators) = @abstractmethod

update!(op::Operators,args...) = @abstractmethod

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod
state_size(f::Iterables) = size(get_state(f))

abstract type FilterCache end

struct Filter{A<:Operators,B<:Iterables,C<:FilterCache} 
  operators::A
  history::Vector{B}
  cache::C
end

function Filter(op::Operators) 
  history = A[]
  cache = allocate_cache(op)
  Filter(op,history,cache)
end

allocate_iterables(f::Filter) = allocate_iterables(f.operators)
allocate_cache(f::Filter) = allocate_cache(f.operators)

function predict!(i::Iterables,f::Filter,args...)
  update!(f.operators,args...)
  predict!(i,f.cache,f.operators,args...)
  return i
end

function update!(i::Iterables,f::Filter,args...)
  update!(i,f.cache,f.operators,args...)
  return i
end

function evaluate!(i::Iterables,f::Filter,args...)
  predict!(i,f,args...)
  update!(i,f,args...)
  push!(f.history,deepcopy(i))
  return get_state(f)
end

function evaluate(f::Filter,args...)
  i = allocate_iterables(f)
  evaluate!(i,f,args...)
  return i
end

(f::Filter)(args...) = evaluate(f,args...)

# utils 

_to_ref(a) = a 
_to_ref(a::Number) = Ref(a)

_from_ref(a) = a 
_from_ref(a::Base.RefValue) = a[]

_set_ref!(a,b) = copyto!(a,b)
_set_ref!(a::Base.RefValue,b::Number) = (a[] = b)