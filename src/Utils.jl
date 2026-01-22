const FType = Union{Function,Map}
const InType = Union{Number,AbstractArray{<:Number}}

jac(f,x::InType) = @abstractmethod
jac(f::Broadcasting{<:Function},x::InType) = jacobian(y -> f.f.(y),x)
jac(f::Function,x::InType) = jacobian(f,x)

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
  mortar(map(similar_mean,v,n))
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