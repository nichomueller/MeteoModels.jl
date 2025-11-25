abstract type Operators end

state_size(op::Operators) = @abstractmethod
measurement_size(op::Operators) = @abstractmethod

allocate_iterables(op::Operators) = @abstractmethod
allocate_cache(op::Operators) = @abstractmethod

update!(op::Operators,args...) = op

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod
state_size(f::Iterables) = size(get_state(f))

Base.copy(i::Iterables) = @abstractmethod

abstract type FilterCache end

struct Filter{A<:Operators,B<:FilterCache} 
  operators::A
  cache::B
end

function Filter(op::Operators) 
  cache = allocate_cache(op)
  Filter(op,cache)
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
  return i
end

function evaluate(f::Filter,args...)
  i = allocate_iterables(f)
  evaluate!(i,f,args...)
  return i
end

(f::Filter)(args...) = evaluate(f,args...)

struct IterableFilter
  filter::Filter 
  initial_state::Iterables
  observation_law::Function 
  observation_grid::AbstractVector 
end

function get_observation_at(it::IterableFilter,k) 
  tk = it.observation_grid[k+1]
  yk = it.observation_law(tk)
  Observation(tk,yk)
end

function update_observation!(y::Observation,it::IterableFilter,k) 
  tk = it.observation_grid[k]
  yk = it.observation_law(tk)
  update!(y,tk,yk)
  return y
end

function Base.iterate(it::IterableFilter)
  if isempty(it.observation_grid)
    return nothing 
  end 

  k = 1
  curstate = it.initial_state
  nextstate = copy(curstate)
  curobs = get_observation_at(it,k)

  evaluate!(nextstate,it.filter,curobs)
  state = (nextstate,curstate,curobs,k+1)

  return nextstate,state
end

function Base.iterate(it::IterableFilter,state)
  curstate,nextstate,curobs,k = state 

  if k > length(it.observation_grid) 
    return nothing 
  end 

  update_observation!(curobs,it,k)
  evaluate!(nextstate,it.filter,curobs)
  state = (nextstate,curstate,curobs,k+1)

  return nextstate,state
end

# utils 

_to_ref(a) = a 
_to_ref(a::Number) = Ref(a)

_from_ref(a) = a 
_from_ref(a::Base.RefValue) = a[]

_set_ref!(a,b) = copyto!(a,b)
_set_ref!(a::Base.RefValue,b::Number) = (a[] = b)