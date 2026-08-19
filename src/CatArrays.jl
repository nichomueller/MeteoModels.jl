struct CatArray{T,N,D} <: AbstractArray{T,N}
  data::Array{T,N}
  ptrs::NTuple{D,Vector{Int}}
end

function vertcat(a::AbstractVector{<:AbstractArray{T,N}}) where {T,N}
  @check !isempty(a) "Cannot vertcat empty collection"
  ptrs = zeros(Int,length(a)+1)
  for i in eachindex(a)
    ptrs[i+1] = size(a[i],1)
  end
  length_to_ptrs!(ptrs)
  s = size(first(a))[2:end]
  data = Array{T,N}(undef,ptrs[end]-1,s...)
  @inbounds for i in eachindex(a)
    @views data[ptrs[i]:ptrs[i+1]-1,_ncolons(Val{N-1}())...] = a[i]
  end
  CatArray(data,(ptrs,))
end

horcat(a::AbstractVector{<:AbstractArray}) = hcat(a...)

function diagcat(a::AbstractVector{<:AbstractMatrix{T}}) where T
  @check !isempty(a) "Cannot diagcat empty collection"
  rptrs = zeros(Int,length(a)+1)
  cptrs = zeros(Int,length(a)+1)
  for i in eachindex(a)
    rptrs[i+1] = size(a[i],1)
    cptrs[i+1] = size(a[i],2)
  end
  length_to_ptrs!(rptrs)
  length_to_ptrs!(cptrs)
  data = zeros(T,rptrs[end]-1,cptrs[end]-1)
  @inbounds for i in eachindex(a)
    data[rptrs[i]:rptrs[i+1]-1,cptrs[i]:cptrs[i+1]-1] = a[i]
  end
  CatArray(data,(rptrs,cptrs))
end

diagcat(a::AbstractMatrix...) = diagcat(collect(a))

function blockmat(C::AbstractMatrix{<:AbstractMatrix{T}}) where T
  nrows,ncols = size(C)
  rptrs = zeros(Int,nrows+1)
  cptrs = zeros(Int,ncols+1)
  for i in 1:nrows
    rptrs[i+1] = size(C[i,1],1)
  end
  for j in 1:ncols
    cptrs[j+1] = size(C[1,j],2)
  end
  length_to_ptrs!(rptrs)
  length_to_ptrs!(cptrs)
  data = zeros(T,rptrs[end]-1,cptrs[end]-1)
  for i in 1:nrows, j in 1:ncols
    data[rptrs[i]:rptrs[i+1]-1,cptrs[j]:cptrs[j+1]-1] = C[i,j]
  end
  CatArray(data,(rptrs,cptrs))
end

Base.size(A::CatArray) = size(A.data)
Base.axes(A::CatArray) = axes(A.data)
Base.getindex(A::CatArray,i...) = A.data[i...]
Base.getindex(A::CatArray{T,N,D},b::BlockArrays.Block{1}) where {T,N,D} = blocks(A)[b.n[1]]
Base.getindex(A::CatArray{T,2,1},b::BlockArrays.Block{2}) where T = blocks(A)[b.n[1]]
Base.getindex(A::CatArray{T,2,2},b::BlockArrays.Block{2}) where T = blocks(A)[b.n[1],b.n[2]]
Base.setindex!(A::CatArray,v,i...) = (A.data[i...] = v)
Base.IndexStyle(::Type{<:CatArray}) = IndexLinear()
Base.collect(A::CatArray) = CatArray(collect(A.data),A.ptrs)
Base.copy(A::CatArray) = CatArray(copy(A.data),map(copy,A.ptrs))
Base.repeat(A::CatArray{T,N,1},counts::Integer...) where {T,N} = CatArray(repeat(A.data,counts...),A.ptrs)
Base.similar(A::CatArray) = CatArray(similar(A.data),A.ptrs)
Base.similar(A::CatArray,::Type{S}) where S = CatArray(similar(A.data,S),A.ptrs)
Base.similar(A::CatArray,inds::Tuple{Base.OneTo,Vararg{Base.OneTo}}) = CatArray(similar(A.data,map(length,inds)),A.ptrs)
Base.similar(A::CatArray{T,N,D},::Type{S},inds::Tuple{Base.OneTo,Vararg{Base.OneTo}}) where {T,N,D,S} = CatArray(similar(A.data,S,map(length,inds)),A.ptrs)

nblocks(A::CatArray{T,N,1}) where {T,N} = length(A.ptrs[1])-1
nblocks(A::CatArray{T,N,D}) where {T,N,D} = ntuple(d -> length(A.ptrs[d])-1,Val{D}())

function BlockArrays.blocks(A::CatArray{T,N,1}) where {T,N}
  [view(A.data,A.ptrs[1][i]:A.ptrs[1][i+1]-1,_ncolons(Val{N-1}())...) for i in 1:nblocks(A)]
end

function BlockArrays.blocks(A::CatArray{T,2,2}) where T
  n1,n2 = nblocks(A)
  [view(A.data,A.ptrs[1][i]:A.ptrs[1][i+1]-1,A.ptrs[2][j]:A.ptrs[2][j+1]-1) for i in 1:n1, j in 1:n2]
end

function _check_ptrs(A::CatArray,B::CatArray)
  @check A.ptrs == B.ptrs "CatArrays have incompatible ptrs"
end

Base.:-(A::CatArray{T,N,1}) where {T,N} = CatArray(-A.data,A.ptrs)

function Base.:+(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  CatArray(A.data+B.data,A.ptrs)
end

Base.:+(A::CatArray{T,N,1},B::AbstractArray) where {T,N} = CatArray(A.data+B,A.ptrs)
Base.:+(A::AbstractArray,B::CatArray{T,N,1}) where {T,N} = CatArray(A+B.data,B.ptrs)

function Base.:-(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  CatArray(A.data-B.data,A.ptrs)
end

Base.:-(A::CatArray{T,N,1},B::AbstractArray) where {T,N} = CatArray(A.data-B,A.ptrs)
Base.:-(A::AbstractArray,B::CatArray{T,N,1}) where {T,N} = CatArray(A-B.data,B.ptrs)

Base.:*(α::Number,A::CatArray{T,N,1}) where {T,N} = CatArray(α*A.data,A.ptrs)
Base.:*(A::CatArray{T,N,1},α::Number) where {T,N} = CatArray(A.data*α,A.ptrs)
Base.:/(A::CatArray{T,N,1},α::Number) where {T,N} = CatArray(A.data/α,A.ptrs)
Base.:\(α::Number,A::CatArray{T,N,1}) where {T,N} = CatArray(α\A.data,A.ptrs)

Base.:*(A::CatArray{T,N,1},B::AbstractMatrix) where {T,N} = CatArray(A.data*B,A.ptrs)
Base.:*(A::AbstractMatrix,B::CatArray{T,N,1}) where {T,N} = A*B.data

function Base.:*(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  CatArray(A.data*B.data,A.ptrs)
end

Base.:*(A::Adjoint{<:Any,<:CatArray},B::CatArray) = A.parent.data'*B.data
Base.:*(A::Transpose{<:Any,<:CatArray},B::CatArray) = transpose(A.parent.data)*B.data
Base.:*(A::CatArray,B::Adjoint{<:Any,<:CatArray}) = A.data*B.parent.data'
Base.:*(A::CatArray,B::Transpose{<:Any,<:CatArray}) = A.data*transpose(B.parent.data)
Base.:*(A::CatArray{T,N,1},B::Adjoint{<:Any,<:CatArray}) where {T,N} = A.data*B.parent.data'
Base.:*(A::CatArray{T,N,1},B::Transpose{<:Any,<:CatArray}) where {T,N} = A.data*transpose(B.parent.data)

Base.:\(A::AbstractMatrix,B::CatArray{T,N,1}) where {T,N} = CatArray(A\B.data,B.ptrs)
Base.:\(A::CatArray{T,N,1},B::AbstractArray) where {T,N} = CatArray(A.data\B,A.ptrs)

function Base.:\(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  CatArray(A.data\B.data,A.ptrs)
end

function LinearAlgebra.mul!(C::CatArray{T,N,1},A::CatArray{T,N,1},B::AbstractMatrix) where {T,N}
  _check_ptrs(C,A)
  mul!(C.data,A.data,B)
  C
end

function LinearAlgebra.mul!(C::CatArray{T,N,1},A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(C,A); _check_ptrs(A,B)
  mul!(C.data,A.data,B.data)
  C
end

function LinearAlgebra.mul!(C::CatArray{T,N,1},A::CatArray{T,N,1},B::AbstractMatrix,α::Number,β::Number) where {T,N}
  _check_ptrs(C,A)
  mul!(C.data,A.data,B,α,β)
  C
end

function LinearAlgebra.mul!(C::CatArray{T,N,1},A::CatArray{T,N,1},B::CatArray{T,N,1},α::Number,β::Number) where {T,N}
  _check_ptrs(C,A); _check_ptrs(A,B)
  mul!(C.data,A.data,B.data,α,β)
  C
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{<:Any,<:CatArray},B::CatArray)
  mul!(C,A.parent.data',B.data)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Transpose{<:Any,<:CatArray},B::CatArray)
  mul!(C,transpose(A.parent.data),B.data)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::CatArray,B::Adjoint{<:Any,<:CatArray},α::Number,β::Number)
  mul!(C,A.data,B.parent.data',α,β)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::CatArray,B::Transpose{<:Any,<:CatArray},α::Number,β::Number)
  mul!(C,A.data,transpose(B.parent.data),α,β)
end

function LinearAlgebra.lmul!(A::AbstractMatrix,B::CatArray{T,N,1}) where {T,N}
  lmul!(A,B.data)
  B
end

function LinearAlgebra.rmul!(A::CatArray{T,N,1},B::AbstractMatrix) where {T,N}
  rmul!(A.data,B)
  A
end

function LinearAlgebra.rmul!(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  rmul!(A.data,B.data)
  A
end

function LinearAlgebra.ldiv!(A::AbstractMatrix,B::CatArray{T,N,1}) where {T,N}
  ldiv!(A,B.data)
  B
end

function LinearAlgebra.ldiv!(A::CatArray{T,N,1},B::CatArray{T,N,1}) where {T,N}
  _check_ptrs(A,B)
  ldiv!(A.data,B.data)
  B
end

function LinearAlgebra.rdiv!(A::CatArray{T,N,1},B::AbstractMatrix) where {T,N}
  rdiv!(A.data,B)
  A
end

function LinearAlgebra.axpy!(α::Number,x::CatArray{T,N,1},y::CatArray{T,N,1}) where {T,N}
  _check_ptrs(x,y)
  axpy!(α,x.data,y.data)
  y
end

function LinearAlgebra.axpby!(α::Number,x::CatArray{T,N,1},β::Number,y::CatArray{T,N,1}) where {T,N}
  _check_ptrs(x,y)
  axpby!(α,x.data,β,y.data)
  y
end

function sample_mean(M::CatArray{T,2}) where T
  CatArray(vec(mean(M.data,dims=2)),(M.ptrs[1],))
end

function allocate_cov(M::CatArray)
  ax = axes(M,1)
  CatArray(similar(M.data,(ax,ax)),(M.ptrs[1],M.ptrs[1]))
end

function perform_step!(
  xf::CatArray{T,1,1},
  integrator::ODEIntegrator,
  x::CatArray{T,1,1}
  ) where T
  @check nblocks(xf) == nblocks(x) == 2
  μf,uf = blocks(xf)
  μ,u = blocks(x)
  copyto!(integrator.p,μ)
  reinit!(integrator,u;t0=integrator.t,reset_dt=false)
  perform_step!(integrator)
  copyto!(μf,integrator.p)
  copyto!(uf,integrator.u)
  xf
end

function perform_step!(
  xf::CatArray{T,2,1},
  integrators::AbstractVector{<:ODEIntegrator},
  x::CatArray{T,2,1}
  ) where T
  @check nblocks(xf) == nblocks(x) == 2
  μf,uf = blocks(xf)
  μ,u = blocks(x)
  @inbounds for (μfi,ufi,integrator,μi,ui) in zip(eachcol(μf),eachcol(uf),integrators,eachcol(μ),eachcol(u))
    copyto!(integrator.p,μi)
    reinit!(integrator,ui;t0=integrator.t,reset_dt=false)
    perform_step!(integrator)
    copyto!(μfi,integrator.p)
    copyto!(ufi,integrator.u)
  end
  xf
end

function perform_step!(
  xf::CatArray{T,2,1},
  cache::PDECache,
  sol::ODEParamSolution,
  x::CatArray{T,2,1}
  ) where T
  @check nblocks(xf) == nblocks(x) == 2
  μf,vf = blocks(xf)
  μ,v = blocks(x)
  @unpack r0,state0,statef,uf,odecache = cache
  to_realisation!(r0,μ)
  to_state!(state0,v)
  cacheit = (r0,state0,statef,uf,odecache)
  (rf,uf),cacheitf = iterate(sol,cacheit)
  update!(cache,cacheitf)
  matrix_of_params!(μf,rf)
  matrix_of_values!(vf,uf)
  xf
end

function to_constraint(c::BlockConstraint,x::CatArray{<:Any,1,1})
  @check length(blocks(c)) == length(blocks(x)) "Incorrect block layout"
  constr(args...) = @abstractmethod
  constr(c::ConstrainTo,x) = c
  constr(c::NoConstraint,x) = ConstrainTo(fill(-Inf,length(x)),fill(Inf,length(x)))
  joint_constraint(map(constr,blocks(c),blocks(x)))
end
