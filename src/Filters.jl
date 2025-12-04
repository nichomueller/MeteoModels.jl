abstract type Filter end

get_prior(f::Filter) = @abstractmethod

get_observation_prior(f::Filter) = @abstractmethod

get_transition_model(f::Filter) = @abstractmethod

get_observation_model(f::Filter) = @abstractmethod

kalman_gain!(f::Filter,posterior::Distribution) = @abstractmethod

update!(posterior::Distribution,f::Filter,args...) = @abstractmethod

get_state(f::Filter) = get_state(get_prior(f))

allocate_distribution(f::Filter) = copy(get_prior(f))

state_size(f::Filter) = dimension(get_prior(f))

observation_size(f::Filter) = dimension(get_observation_prior(f))

function transition!(posterior::Distribution,f::Filter)
  prior = get_prior(f)
  evaluate!(posterior,get_transition_model(f),prior)
end

function observation!(f::Filter,posterior::Distribution)
  obs_prior = get_observation_prior(f)
  evaluate!(obs_prior,get_transition_model(f),posterior)
end

function innovation!(f::Filter,z::InType)
  obs_prior = get_observation_prior(f)
  innovation!(z,obs_prior)
end

function forecast!(posterior::Distribution,f::Filter)
  transition!(posterior,f)
end

function analyse!(posterior::Distribution,f::Filter,args...)
  observation!(f,posterior)
  kalman_gain!(f,posterior)
  ỹ = innovation!(f,args...)
  update!(posterior,f,ỹ)
end

function evaluate!(posterior::Distribution,f::Filter,args...)
  prior = get_prior(f)
  copyto!(posterior,prior)
  forecast!(posterior,f)
  analyse!(posterior,f,args...)
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
  obs = get_observation_model(f)
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,length(grid))

  for (k,δk) in enumerate(grid)
    yδk = realization(obs,δk)
    evaluate!(posterior,f,yδk)
    history[k] = copy(posterior)
  end 

  return history
end

function loop(f::Filter,grid::AbstractVector,obs_generator::Function)
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,length(grid))

  for (k,δk) in enumerate(grid)
    yδk = obs_generator(δk)
    evaluate!(posterior,f,yδk)
    history[k] = copy(posterior)
  end 

  return history
end

# function visualize(history::AbstractVector{<:Distribution})
  
# end

# utils 

function innovation!(z::Number,y::Distribution)
  z - get_state(y)
end

function innovation!(z::AbstractArray,y::Distribution)
  z .-= get_state(y)
  z
end

