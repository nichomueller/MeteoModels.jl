const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray{<:Number}}

# helpers for jacobians 

jac(f,x::InType) = @abstractmethod
jac(f::Broadcasting{<:Function},x::InType) = jacobian(y -> f.f.(y),x)
jac(f::Function,x::InType) = jacobian(f,x)

# helpers for distributions  

dimension(v::Number) = 1
dimension(v::AbstractVector) = length(v)
dimension(v::BlockVector) = map(dimension,blocks(v))

function allocate_mean(n::Int)
  zeros(Float64,n)
end

function allocate_mean(n::AbstractVector)
  mortar(map(allocate_mean,n))
end

function allocate_cov(n::Int)
  diagm(rand(Float64,n))
end

function allocate_cov(n::AbstractVector)
  BlockDiagonal(map(allocate_cov,n))
end

function allocate_values(n::Int,ncol::Int)
  zeros(Float64,n,ncol)
end

function allocate_values(n::AbstractVector,ncol::Int)
  block_cat(map(x -> allocate_values(x,ncol),n))
end

function similar_mean(v::AbstractVector,n::Int=length(v))
  similar(v,n)
end

function similar_mean(v::BlockVector,n::AbstractVector=map(length,blocks(v)))
  mortar(map(similar_mean,blocks(v),n))
end

function similar_cov(v::AbstractVector,n::Int=length(v))
  T = eltype(v)
  diagm(rand(T,n))
end

function similar_cov(v::BlockVector,n::AbstractVector=map(length,blocks(v)))
  BlockDiagonal(map(similar_cov,blocks(v),n))
end

function similar_values(v::AbstractVector,ncol::Int,n::Int=length(v))
  T = eltype(v)
  zeros(T,n,ncol)
end

function similar_values(v::BlockVector,ncol::Int,n::AbstractVector=map(length,blocks(v)))
  block_cat(map((x,y) -> similar_values(x,ncol,y),blocks(v),n))
end

function block_cat(v::AbstractVector{A}) where A<:AbstractMatrix
  m = Matrix{A}(undef,length(v),1)
  for i in eachindex(v)
    m[i] = v[i]
  end
  mortar(m)
end

# helpers for passing from MeteoModels types to Gridap/GridapROMs types

param_dimension(p::ParamSpace) = length(p.param_domain)
param_dimension(p::TransientParamSpace) = param_dimension(p.parametric_space)

matrix_of_params(r::AbstractRealization) = RBSteady._get_params_marix(r)

function to_realization(param::AbstractMatrix,r̃::Realization)
  Realization(eachcol(param))
end
 
function to_realization(param::AbstractMatrix,r̃::GenericTransientRealization)
  r = to_realization(param,get_params(r̃))
  GenericTransientRealization(r,r̃.times,r̃.t0)
end

function to_realization(param::AbstractMatrix,r̃::TransientRealizationAt)
  r = to_realization(param,get_params(r̃))
  TransientRealizationAt(r,r̃.t)
end

matrix_of_values(u::ConsecutiveParamVector) = get_all_data(u)
matrix_of_values(u::RBParamVector) = get_all_data(u.fe_data)

function to_param_array(vals::AbstractMatrix,ũ::ConsecutiveParamVector)
  ConsecutiveParamArray(vals)
end

function to_param_array(vals::AbstractMatrix,ũ::RBParamVector)
  fe_data = to_param_array(vals,ũ.fe_data)
  RBParamVector(ũ.data,fe_data)
end

function to_state(vals::AbstractMatrix,state::NTuple{N,T},::ThetaMethod) where {N,T<:AbstractParamVector}
  ntuple(i -> to_param_array(vals,state[i]),Val(N))
end