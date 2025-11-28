abstract type Operator end

state_size(op::Operator) = @abstractmethod
measurement_size(op::Operator) = @abstractmethod

allocate_iterables(op::Operator) = @abstractmethod
allocate_cache(op::Operator) = @abstractmethod

update!(op::Operator,args...) = op

abstract type Iterables end

get_state(i::Iterables) = @abstractmethod
state_size(f::Iterables) = size(get_state(f))

Base.copy(i::Iterables) = @abstractmethod

jacobian(a::Model,i::Iterables) = jacobian(a,get_state(i))
discretize(a::GenericModel,i::Iterables) = discretize(a,get_state(i))

abstract type FilterCache end

struct Filter{A<:Operator,B<:Iterables,C<:FilterCache} 
  operators::A
  iterables::B
  cache::C
end

function Filter(op::Operator,i::Iterables) 
  cache = allocate_cache(op)
  Filter(op,i,cache)
end

function Filter(op::Operator) 
  i = allocate_iterables(op)
  Filter(op,i)
end

allocate_iterables(f::Filter) = copy(f.iterables)
allocate_cache(f::Filter) = allocate_cache(f.operators)

function predict!(i::Iterables,f::Filter,args...)
  update!(f.operators,i,f.cache,args...)
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

function loop(f::Filter{A,B},obs_law::Function,grid::AbstractVector) where {A,B}
  iter = f.iterables
  history = Vector{B}(undef,length(grid))

  for tk in grid
    yk = Observation(tk,obs_law(tk))
    evaluate!(iter,f,yk)
    push!(history,copy(iter))
  end 

  return history
end