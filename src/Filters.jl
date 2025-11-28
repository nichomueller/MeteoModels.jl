abstract type Filter end

get_prior(f::Filter) = @abstractmethod

get_state(f::Filter) = get_state(get_prior(f))

allocate_distribution(f::Filter) = copy(get_prior(f))

realization(f::Filter,args...) = realization(get_prior(f),args...)

state_size(f::Filter) = dimension(get_prior(f))

get_measurement_model(f::Filter) = @abstractmethod

measurement_size(f::Filter) = dimension(get_measurement_model(f))

linearize(f::Filter) = linearize(f,get_prior(f))

function update!(posterior::Distribution,f::Filter,args...)
  @abstractmethod
end

function predict!(posterior::Distribution,f::Filter,args...)
  @abstractmethod
end

function evaluate!(posterior::Distribution,f::Filter,args...)
  prior = get_prior(f)
  copyto!(posterior,prior)
  predict!(posterior,f,args...)
  update!(posterior,f,args...)
  copyto!(prior,posterior)
  return posterior
end

function evaluate(f::Filter,args...)
  d = allocate_distribution(f)
  evaluate!(d,f,args...)
  return d
end

(f::Filter)(args...) = evaluate(f,args...)

function loop(f::Filter,grid::AbstractVector)
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,length(grid))

  for δ in grid
    yδ = realization(f,δ)
    evaluate!(posterior,f,yδ)
    push!(history,copy(posterior))
  end 

  return history
end