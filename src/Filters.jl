abstract type Filter end

get_prior(f::Filter) = @abstractmethod

get_observation_prior(f::Filter) = @abstractmethod

get_transition_model(f::Filter) = @abstractmethod

get_observation_model(f::Filter) = @abstractmethod

transition!(posterior::Distribution,f::Filter) = @abstractmethod

observation!(f::Filter,posterior::Distribution) = @abstractmethod

kalman_gain!(f::Filter,posterior::Distribution) = @abstractmethod

mixed_cov!(K::AbstractMatrix,f::Filter,posterior::Distribution) = @abstractmethod

update!(posterior::Distribution,f::Filter,args...) = @abstractmethod

get_state(f::Filter) = get_state(get_prior(f))

allocate_distribution(f::Filter) = copy(get_prior(f))

state_size(f::Filter) = dimension(get_prior(f))

observation_size(f::Filter) = dimension(get_observation_prior(f))

function innovation!(f::Filter,z::InType)
  obs_prior = get_observation_prior(f)
  innovation!(obs_prior,z)
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

function loop(f::Filter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    evaluate!(posterior,f,yk)
    history[k] = copy(posterior)
  end 

  return history
end

abstract type FunctionFilter <: Filter end

evaluate(f::FunctionFilter,args...) = @abstractmethod

function loop(f::FunctionFilter,obs::AbstractArray{T,N}) where {T,N} 
  posterior = allocate_distribution(f)
  history = Vector{typeof(posterior)}(undef,size(obs,N))

  for k in axes(obs,N)
    yk = selectdim(obs,N,k)
    evaluate!(posterior,f(k),yk)
    history[k] = copy(posterior)
  end 

  return history
end

function visualize(
  history::AbstractVector{<:Distribution},
  grid=eachindex(history);
  index::Int=1
  )

  μ = map(get_state,history)
  σ² = map(get_cov,history)

  μᵢ = map(x -> getindex(x,index),μ)
  σᵢ = map(x -> sqrt(getindex(x,index,index)),σ²)
  plot(grid,μᵢ,label="Prediction",color=:red,linewidth=3,ribbon=σᵢ,fillcolor=:blue,fillalpha=0.3)
end

function visualize(
  true_values::AbstractMatrix,
  history::AbstractVector{<:Distribution},
  grid=eachindex(history);
  index::Int=1
  )

  visualize(history,grid;index)
  plot!(grid,true_values[index,:],color=:black,linewidth=3,label="True state")
end

# utils 

function innovation!(d::Distribution,z::InType)
  ỹ = _innovation!(d,z)
  ỹ .*= -1
  ỹ
end

function _innovation!(d::Distribution,z::InType)
  y = get_state(d)
  y .-= z
  y
end

function _innovation!(d::Ensemble,z::AbstractArray)
  y = get_state(d)
  @inbounds @views for i in 1:ensemble_size(d)
    y[:,i] .-= z 
  end
  y
end
