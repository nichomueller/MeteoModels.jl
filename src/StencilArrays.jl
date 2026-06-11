struct TimeStencils
  all_grid::AbstractVector
  warmup_grid::AbstractVector
  train_grid::AbstractVector
  washout_grid::AbstractVector
  da_grid::AbstractVector
  all_obs_grid::AbstractVector
  da_obs_grid::AbstractVector
end

function TimeStencils(;dt,dt_obs=dt,t0=0.0,t_warmup=0.0,t_train=0.0,t_wash=0.0,t_da)
  t0_warmup = t0
  tf_warmup = t0_warmup + t_warmup
  t0_tv = tf_warmup
  tf_tv = t0_tv + t_train
  t0_wash = tf_tv
  tf_wash = t0_wash + t_wash
  t0_da = tf_wash
  tf_da = t0_da + t_da

  TimeStencils(
    stencil((t0,tf_da),dt),
    stencil((t0_warmup,tf_warmup),dt_obs),
    stencil((t0_tv,tf_tv),dt_obs),
    stencil((t0_wash,tf_wash),dt_obs),
    stencil((t0_da,tf_da),dt),
    stencil((t0,tf_da),dt_obs),
    stencil((t0_da,tf_da),dt_obs)
  )
end

const ALL = 0  
const WARMUP = 1  
const TRAIN = 2
const WASHOUT = 3
const DA = 4
const ALLOBS = 5 
const DAOBS = 6
const PHASES = (ALL,WARMUP,TRAIN,WASHOUT,DA,ALLOBS,DAOBS)

function phase2symbol(phase::Int)
  if phase == ALL
    return :all_grid
  elseif phase == WARMUP
    return :warmup_grid
  elseif phase == TRAIN
    return :train_grid
  elseif phase == WASHOUT
    return :washout_grid
  elseif phase == DA
    return :da_grid
  elseif phase == ALLOBS
    return :all_obs_grid
  elseif phase == DAOBS
    return :da_obs_grid
  else
    @notimplemented "Invalid phase"
  end
end

Base.getindex(s::TimeStencils,phase::Int) = getproperty(s,phase2symbol(phase))

struct StencilArray{A<:AbstractArray}
  array::A
  stencils::TimeStencils
  phase::Int
end

function from_stencil(a::StencilArray,newphase::Int=a.phase)
  old_stencil = a.stencils[a.phase]
  new_stencil = a.stencils[newphase]
  new_stencil == old_stencil && return a.array
  new_stencil ⊆ old_stencil && return restrict(a.array,old_stencil,new_stencil)
  old_stencil ⊆ new_stencil && return expand(a.array,old_stencil,new_stencil)
  @notimplemented "The new stencil must be a subset or superset of the old one"
end

function to_stencil(x::AbstractArray,s::TimeStencils,phase::Int=ALL)
  @check 0 <= phase <= length(PHASES) "Invalid phase"
  StencilArray(x,s,phase)
end

Base.getindex(a::StencilArray,phase::Int) = from_stencil(a,phase)

function restrict(
  fine_vals::AbstractArray{T,N},
  fine_stencil::AbstractVector,
  coarse_stencil::AbstractVector,
  ) where {T<:Number,N}

  @check length(fine_stencil) == size(fine_vals,N)
  coarse_size = (size(fine_vals)[1:end-1]...,length(coarse_stencil))
  coarse_vals = zeros(T,coarse_size...)
  fine_slices = eachslice(fine_vals,dims=N)
  coarse_slices = eachslice(coarse_vals,dims=N)
  count = 0
  for i in eachindex(fine_stencil)
    count == length(coarse_stencil) && break
    if coarse_stencil[count+1] ≈ fine_stencil[i]
      coarse_slices[count+1] .= fine_slices[i] 
      count += 1
    end
  end
  @check count == length(coarse_stencil) "The coarse stencil must be a subset of the fine one"
  return coarse_vals
end

function restrict(
  fine_vals::AbstractVector{T},
  fine_stencil::AbstractVector,
  coarse_stencil::AbstractVector,
  ) where T

  @check length(fine_stencil) == length(fine_vals)
  coarse_len = length(coarse_stencil)
  coarse_vals = Vector{T}(undef,coarse_len)
  count = 0
  for i in eachindex(fine_stencil)
    count == length(coarse_stencil) && break
    if coarse_stencil[count+1] ≈ fine_stencil[i]
      coarse_vals[count+1] = fine_vals[i] 
      count += 1
    end
  end
  @check count == length(coarse_stencil) "The coarse stencil must be a subset of the fine one"
  return coarse_vals
end

function expand(
  coarse_vals::AbstractArray{T,N},
  coarse_stencil::AbstractVector,
  fine_stencil::AbstractVector,
  ) where {T<:AbstractFloat,N}
  
  @check length(coarse_stencil) == size(coarse_vals,N)
  fine_size = (size(coarse_vals)[1:end-1]...,length(fine_stencil))
  fine_vals = zeros(T,fine_size...)
  fill!(fine_vals,NaN)
  fine_slices = eachslice(fine_vals,dims=N)
  coarse_slices = eachslice(coarse_vals,dims=N)
  count = 0
  for i in eachindex(fine_stencil)
    count == length(coarse_stencil) && break
    if coarse_stencil[count+1] ≈ fine_stencil[i]
      fine_slices[i] .= coarse_slices[count+1] 
      count += 1
    end
  end
  @check count == length(coarse_stencil) "The coarse stencil must be a subset of the fine one"
  return fine_vals
end

function expand(
  coarse_vals::AbstractVector{T},
  coarse_stencil::AbstractVector,
  fine_stencil::AbstractVector,
  ) where T
  
  @check length(coarse_stencil) == length(coarse_vals)
  fine_size = (length(fine_stencil),)
  fine_vals = Vector{T}(undef,fine_size...)
  count = 0
  for i in eachindex(fine_stencil)
    count == length(coarse_stencil) && break
    if coarse_stencil[count+1] ≈ fine_stencil[i]
      fine_vals[i] = coarse_vals[count+1] 
      count += 1
    end
  end
  @check count == length(coarse_stencil) "The coarse stencil must be a subset of the fine one"
  return fine_vals
end

# utils 

function stencil(s::AbstractVector)
  s
end

function stencil(limits::Tuple{Real,Real},dt::Real)
  a,b = limits
  N = round(Int,(b-a)/dt) 
  a .+ (1:N) .* dt
end