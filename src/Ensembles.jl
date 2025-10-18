struct Ensemble{T,N,F<:Function} <: AbstractArray{T,N}
  data::Array{T,N}
  obs_fun::F
end

function Ensemble(
  data::AbstractArray,
  obs_matrix::AbstractMatrix{T},
  obs_vector::AbstractVector=zeros(T,size(obs_matrix,1))
  ) where T

  obs_fun = x -> obs_matrix*x + obs_vector
  Ensemble(data,obs_fun)
end

const EnsembleMatrix{T,F} = Ensemble{T,2,F}

Base.size(a::Ensemble) = size(a.data)
Base.getindex(a::Ensemble{T,N},i::Vararg{Int,N}) where {T,N} = getindex(a.data,i)

abstract type EnsembleFilter <: Filter end

struct EnKF <: EnsembleFilter
  err_matrix
end