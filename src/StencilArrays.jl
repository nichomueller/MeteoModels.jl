"""
    struct TimeStencils

Descriptor of the time axis partitioned into named phases used by the RC/DA pipeline.

Fields:
- `dt`: model integration time step;
- `dt_obs`: observation time step (defaults to `dt`);
- `all_window`, `warmup_window`, `train_window`, `washout_window`, `spread_window`, `da_window`:
  `(t_start, t_end)` tuples for each phase.

Construct via keyword arguments:
```julia
ts = TimeStencils(dt=0.01, dt_obs=0.1, t_warmup=10.0, t_train=50.0, t_wash=5.0, t_spread=5.0, t_da=20.0)
```
Index with phase constants to get the corresponding time stencil:
```julia
ts[DA]      # time steps in the DA phase
ts[TRAIN]   # time steps in the training phase
```
"""
struct TimeStencils
  dt::Real
  dt_obs::Real
  all_window::Tuple{Real,Real}
  warmup_window::Tuple{Real,Real}
  train_window::Tuple{Real,Real}
  washout_window::Tuple{Real,Real}
  spread_window::Tuple{Real,Real}
  da_window::Tuple{Real,Real}
end

function TimeStencils(;dt,dt_obs=dt,t0=0.0,t_warmup=0.0,t_train=0.0,t_wash=0.0,t_spread=0.0,t_da)
  t0_warmup = t0
  tf_warmup = t0_warmup + t_warmup
  t0_tv = tf_warmup
  tf_tv = t0_tv + t_train
  t0_wash = tf_tv
  tf_wash = t0_wash + t_wash
  t0_spread = tf_wash
  tf_spread = t0_spread + t_spread
  t0_da = tf_spread
  tf_da = t0_da + t_da

  TimeStencils(
    dt,dt_obs,
    (t0,tf_da),
    (t0_warmup,tf_warmup),
    (t0_tv,tf_tv),
    (t0_wash,tf_wash),
    (t0_spread,tf_spread),
    (t0_da,tf_da)
  )
end

"""
    ALL, WARMUP, TRAIN, WASHOUT, SPREAD, DA
    OBSALL, OBSWARMUP, OBSTRAIN, OBSWASHOUT, OBSSPREAD, OBSDA

Integer phase tags used to index a [`TimeStencils`](@ref) descriptor or a
[`StencilArray`](@ref).  The `OBS*` variants select the same time window but
at the observation cadence (`dt_obs`) rather than the model cadence (`dt`).

| Constant   | Phase                          |
|------------|-------------------------------|
| `ALL`      | entire simulation window      |
| `WARMUP`   | reservoir warm-up             |
| `TRAIN`    | RC training                   |
| `WASHOUT`  | washout (transient discard)   |
| `SPREAD`   | ensemble spread phase         |
| `DA`       | data-assimilation window      |
"""
const ALL = 0
const WARMUP = 1
const TRAIN = 2
const WASHOUT = 3
const SPREAD = 4
const DA = 5
const OBSALL = 6
const OBSWARMUP = 7
const OBSTRAIN = 8
const OBSWASHOUT = 9
const OBSSPREAD = 10
const OBSDA = 11

const PHASES = (
  ALL,
  WARMUP,
  TRAIN,
  WASHOUT,
  SPREAD,
  DA,
  OBSALL,
  OBSWARMUP,
  OBSTRAIN,
  OBSSPREAD,
  OBSWASHOUT,
  OBSDA
)

function phase2symbol(phase::Int)
  if phase == ALL || phase == OBSALL
    return :all_window
  elseif phase == WARMUP || phase == OBSWARMUP
    return :warmup_window
  elseif phase == TRAIN || phase == OBSTRAIN
    return :train_window
  elseif phase == WASHOUT || phase == OBSWASHOUT
    return :washout_window
  elseif phase == SPREAD || phase == OBSSPREAD
    return :spread_window
  elseif phase == DA || phase == OBSDA
    return :da_window
  else
    @notimplemented "Invalid phase"
  end
end

function Base.getindex(s::TimeStencils,phase::Int)
  sym = phase2symbol(phase)
  step = phase > DA ? s.dt_obs : s.dt
  stencil(getproperty(s,sym),step)
end

function Base.getindex(s::TimeStencils,phases::UnitRange)
  @check !isempty(phases) "Empty phase range"
  @check all(p -> p > DA,phases) || all(p -> p ≤ DA,phases) "Cannot transition from a regular phase to an obs phase"
  step = first(phases) > DA ? s.dt_obs : s.dt
  t_start = first(getproperty(s,phase2symbol(first(phases))))
  t_end   = last(getproperty(s,phase2symbol(last(phases))))
  stencil((t_start,t_end),step)
end

"""
    struct StencilArray{A<:AbstractArray,B}

Wraps an array together with a [`TimeStencils`](@ref) descriptor and a phase tag so
that it can be re-indexed at a different phase (with automatic restriction or expansion).

Construct via [`to_stencil`](@ref); index with a phase constant to extract or rephase the data:
```julia
sa = to_stencil(x, ts, TRAIN)
sa[DA]    # restrict (or expand) x to the DA phase
```
"""
struct StencilArray{A<:AbstractArray,B}
  array::A
  stencils::TimeStencils
  phase::B
end

function from_stencil(a::StencilArray,newphase=a.phase)
  newphase == a.phase && return a.array
  old_stencil = a.stencils[a.phase]
  new_stencil = a.stencils[newphase]
  n_old = length(old_stencil)
  n_new = length(new_stencil)
  n_new <= n_old && return restrict(a.array,old_stencil,new_stencil)
  return expand(a.array,old_stencil,new_stencil)
end

function to_stencil(x::AbstractArray,s::TimeStencils,phase=ALL)
  @check 0 <= first(phase) && last(phase) <= length(PHASES) "Invalid phase"
  StencilArray(x,s,phase)
end

Base.getindex(a::StencilArray,phase) = from_stencil(a,phase)

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

function restrict(s::TransientSnapshots,ts::TimeStencils,param_ids=1:num_params(s),phase=ALL)
  μ = get_realisation(s)
  stimes = get_times(μ)
  rtimes = ts[phase]
  istart = findfirst(t -> t ≈ first(rtimes),stimes)
  iend = findfirst(t -> t ≈ last(rtimes),stimes)
  @check !isnothing(istart) "The start time of the new stencil is not in the original stencil"
  @check !isnothing(iend) "The end time of the new stencil is not in the original stencil"
  select_times(select_snapshots(s,param_ids),istart:iend)
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