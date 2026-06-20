abstract type DecompositionStrategy end

struct DiagonalDecomposition <: DecompositionStrategy end

struct BlockDecomposition <: DecompositionStrategy
  nblocks::Int
end

function decompose(::DecompositionStrategy,A::AbstractMatrix)
  @abstractmethod
end

function decompose(s::DecompositionStrategy,d::SecondMoment)
  decompose(s,cov(d))
end

function decompose(args...;nblocks=1,strategy=BlockDecomposition(nblocks))
  decompose(strategy,args...)
end

struct MatrixDecomposition
  matrices::AbstractVector{<:AbstractMatrix}
  weights::AbstractVector{<:Real}
end

function decompose(s::DiagonalDecomposition,A::AbstractMatrix)
  @check issquare(A)
  n = size(A,2)
  matrices = map(1:n) do i
    d = zeros(n)
    d[i] = 1
    Diagonal(d)
  end
  weights = ones(n)
  MatrixDecomposition(matrices,weights)
end

function decompose(s::BlockDecomposition,A::AbstractMatrix)
  @check issquare(A)
  n = size(A,2)
  m = Int(n / s.nblocks)
  matrices = map(CartesianIndices((s.nblocks,s.nblocks))) do I
    i,j = Tuple(I)
    mat = zeros(size(A))
    for i in (i-1)*m+1:i*m
      for j in (j-1)*m+1:j*m
        mat[i,j] = 1
      end
    end
    mat
  end |> vec
  weights = ones(length(matrices))
  MatrixDecomposition(matrices,weights)
end

function linear_combination!(cache,d::MatrixDecomposition)
  fill!(cache,zero(eltype(cache)))
  @inbounds for i in eachindex(d.matrices)
    axpy!(d.weights[i],d.matrices[i],cache)
  end
  cache
end

function linear_combination(d::MatrixDecomposition)
  cache = similar(first(d.matrices))
  linear_combination!(cache,d)
end

function llsq!(x::AbstractArray,A::AbstractMatrix,b::AbstractArray)
  F = qr!(A)
  ldiv!(x,F,b)
  x
end

function rlsq!(x::AbstractMatrix,b::AbstractMatrix,A::AbstractMatrix)
  llsq!(x',collect(A'),collect(b'))
  x
end

struct MemoCache{T}
  current::T
  previous::T
end

function update!(a::MemoCache{T},x::T) where T
  copyto!(a.previous,a.current)
  copyto!(a.current,x)
  return a
end

struct AdaptiveCache
  trans_cache::MemoCache{<:AbstractMatrix}
  obs_cache::MemoCache{<:AbstractMatrix}
  innov_cache::MemoCache{<:AbstractArray}
  Qdec::MatrixDecomposition
  Qtemp::AbstractMatrix
  Rtemp::AbstractMatrix
  Qadapt::AbstractMatrix
  Radapt::AbstractMatrix
  HF::AbstractMatrix
  HFK::AbstractMatrix
  yy::AbstractMatrix
  Ck::AbstractMatrix
  FΣF::AbstractMatrix
  HFbuf::AbstractMatrix
  Amat::AbstractMatrix
  cvec::AbstractVector
end

function AdaptiveCache(f::KalmanFilter;kwargs...)
  noise = get_noise(f)
  Qdec = decompose(noise;kwargs...)
  prior = get_prior(f)
  obs_prior = get_observation_prior(f)
  n = length(mean(prior))
  m = length(mean(obs_prior))
  p = length(Qdec.weights)
  AdaptiveCache(
    MemoCache(zeros(n,n),zeros(n,n)),
    MemoCache(zeros(m,n),zeros(m,n)),
    MemoCache(zeros(m),zeros(m)),
    Qdec,
    zeros(n,n),zeros(m,m),
    zeros(n,n),zeros(m,m),
    zeros(m,n),zeros(m,m),
    zeros(m,m),zeros(m,m),
    zeros(n,n),zeros(m,n),
    zeros(m^2,p),zeros(m^2)
  )
end

function update_transition_cache!(c::AdaptiveCache,x)
  update!(c.trans_cache,x)
end

function update_observation_cache!(c::AdaptiveCache,x)
  update!(c.obs_cache,x)
end

function update_innovation_cache!(c::AdaptiveCache,x)
  update!(c.innov_cache,x)
end

struct AdaptiveKalmanFilter{F<:KalmanFilter} <: KalmanFilter
  filter::F
  last_posterior::SecondMoment
  step::Real
  cache::AdaptiveCache
end

function AdaptiveKalmanFilter(filter::KalmanFilter;step=0.1,kwargs...)
  last_posterior = copy(get_prior(filter))
  cache = AdaptiveCache(filter;kwargs...)
  AdaptiveKalmanFilter(filter,last_posterior,step,cache)
end

get_prior(f::AdaptiveKalmanFilter) = get_prior(f.filter)
get_observation_prior(f::AdaptiveKalmanFilter) = get_observation_prior(f.filter)
get_transition_model(f::AdaptiveKalmanFilter) = get_transition_model(f.filter)
get_observation_model(f::AdaptiveKalmanFilter) = get_observation_model(f.filter)
get_noise(f::AdaptiveKalmanFilter) = get_noise(f.filter)
get_observation_noise(f::AdaptiveKalmanFilter) = get_observation_noise(f.filter)
get_prior_cache(f::AdaptiveKalmanFilter) = get_prior_cache(f.filter)
get_cache(f::AdaptiveKalmanFilter) = f.cache

function update_transition_cache!(f::AdaptiveKalmanFilter)
  model = get_transition_model(f)
  prior = get_prior(f)
  J = jac(model,mean(prior))
  update_transition_cache!(f.cache,J)
end

function update_observation_cache!(f::AdaptiveKalmanFilter)
  model = get_observation_model(f)
  prior = get_prior(f)
  J = jac(model,mean(prior))
  update_observation_cache!(f.cache,J)
end

function transition!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  update_transition_cache!(f)
  noise = get_noise(f)
  copyto!(cov(noise),f.cache.Qadapt)
  transition!(posterior,f.filter)
end

function observation!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  noise = get_observation_noise(f)
  copyto!(cov(noise),f.cache.Radapt)
  o = observation!(f.filter,posterior)
  update_observation_cache!(f)
  return o
end

function innovation!(f::AdaptiveKalmanFilter,z::InType)
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f.cache,ỹ)
  return ỹ
end

function kalman_gain!(f::AdaptiveKalmanFilter,posterior::SecondMoment)
  kalman_gain!(f.filter,posterior)
end

function update!(posterior::SecondMoment,f::AdaptiveKalmanFilter,ỹ::InType)
  update!(posterior,f.filter,ỹ)
end

function update_cache!(f::AdaptiveKalmanFilter)
  c = f.cache
  Fcur,Fprev = c.trans_cache.current,c.trans_cache.previous
  Hcur,Hprev = c.obs_cache.current,c.obs_cache.previous
  ỹcur,ỹprev = c.innov_cache.current,c.innov_cache.previous

  Kprev = get_kalman_gain(f)
  Σprev = cov(f.last_posterior)
  Σcur = cov(get_prior(f))

  mul!(c.HF,Hcur,Fcur)
  mul!(c.yy,ỹprev,ỹprev')
  mul!(c.HFK,c.HF,Kprev)

  mul!(c.Ck,c.HFK,c.yy)
  mul!(c.Ck,ỹcur,ỹprev',1,1)
  mul!(c.FΣF,Fprev,Σprev)
  mul!(c.Qtemp,c.FΣF,Fprev')
  mul!(c.HFbuf,c.HF,c.Qtemp)
  mul!(c.Ck,c.HFbuf,Hprev',-1,1)

  @inbounds @views for (i,Qp) in enumerate(c.Qdec.matrices)
    mul!(c.HFbuf,c.HF,Qp)
    Ai = reshape(c.Amat[:,i],size(c.Ck))
    mul!(Ai,c.HFbuf,Hprev')
  end

  copyto!(c.cvec,vec(c.Ck))
  llsq!(c.Qdec.weights,c.Amat,c.cvec)
  linear_combination!(c.Qtemp,c.Qdec)

  mul!(c.Rtemp,ỹprev,ỹprev')
  mul!(c.HFbuf,Hprev,Σcur)
  mul!(c.Rtemp,c.HFbuf,Hprev',-1,1)

  @. c.Qadapt += f.step * (c.Qtemp - c.Qadapt)
  @. c.Radapt += f.step * (c.Rtemp - c.Radapt)

  symmetrize!(c.Qadapt)
  symmetrize!(c.Radapt)

  return c
end

function forecast!(posterior::SecondMoment,f::AdaptiveKalmanFilter)
  prior = get_prior(f)
  copyto!(f.last_posterior,prior)
  transition!(posterior,f)
  copyto!(prior,posterior)
  posterior
end

function analyse!(posterior::SecondMoment,f::AdaptiveKalmanFilter,args...)
  observation!(f,posterior)
  ỹ = innovation!(f,args...)
  update_cache!(f)
  kalman_gain!(f,posterior)
  update!(posterior,f,ỹ)
end

function reset!(f::AdaptiveKalmanFilter)
  reset!(f.filter)
end

# nonlinear case: implement the extended Kalman filter (EKF)

function forecast!(
  posterior::SecondMoment,
  f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:NonlinearModel}}
  )

  flin = linearise_around_transition(f)
  forecast!(posterior,flin)
  return posterior
end

function analyse!(
  posterior::SecondMoment,
  f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:Any,<:NonlinearModel}},
  z::InType
  )

  flin = linearise_around_observation(f)
  analyse!(posterior,flin,z)
  return posterior
end

function linearise_around_transition(f::AdaptiveKalmanFilter{<:NonlinearModel})
  flin = linearise_around_transition(f.filter)
  AdaptiveKalmanFilter(flin,f.last_posterior,f.step,f.cache)
end

function linearise_around_observation(f::AdaptiveKalmanFilter{<:GenericKalmanFilter{<:Any,<:NonlinearModel}})
  flin = linearise_around_observation(f.filter)
  AdaptiveKalmanFilter(flin,f.last_posterior,f.step,f.cache)
end

# ensemble case

function update_transition_cache!(f::AdaptiveKalmanFilter{<:EnsembleKalmanFilter})
  xf = get_state(get_prior(f))
  xa = get_state(f.last_posterior)
  _xa = get_state(get_prior_cache(f))
  copyto!(_xa,xa)
  rlsq!(f.cache.FΣF,xf,_xa)
  update_transition_cache!(f.cache,f.cache.FΣF)
end

function update_observation_cache!(f::AdaptiveKalmanFilter{<:EnsembleKalmanFilter})
  x̃f = get_state(get_prior_cache(f))
  copyto!(x̃f,get_state(get_prior(f)))
  add_draw!(x̃f,get_noise(f))
  y = get_state(get_observation_prior(f))
  rlsq!(f.cache.HFbuf,y,x̃f)
  update_observation_cache!(f.cache,f.cache.HFbuf)
end

function innovation!(f::AdaptiveKalmanFilter{<:EnsembleKalmanFilter},z::InType)
  ỹ = innovation!(f.filter,z)
  update_innovation_cache!(f.cache,mean(ỹ))
  return ỹ
end

function loop(f::AdaptiveKalmanFilter,obs::AbstractArray{T,N},args...;kwargs...) where {T,N}
  prior = get_prior(f)
  posterior = copy(prior)
  history = Vector{typeof(posterior)}(undef,size(obs,N))
  table = ResultsTable()

  # 1st iteration: use inner filter directly (no adaptation on first step)
  yk = selectdim(obs,N,1)
  evaluate!(posterior,f.filter,yk)
  update_table!(table,f,yk)
  history[1] = copy(posterior)

  for k in 2:size(obs,N)
    yk = selectdim(obs,N,k)
    copyto!(prior,posterior)
    isnan(yk) ? evaluate!(posterior,f) : evaluate!(posterior,f,yk)
    update_table!(table,f,yk)
    history[k] = copy(posterior)
  end

  reset!(f)
  return FilterResults(history,table)
end