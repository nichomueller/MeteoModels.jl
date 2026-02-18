struct StencilArray{A<:AbstractArray,B<:AbstractVector} 
  array::A
  stencil::B 
end

for (f,_f) in zip((:restrict,:expand),(:_restrict,:_expand))
  @eval begin
    function $f(a::AbstractArray,astencil::AbstractVector,stencil::AbstractVector) 
      $_f(a,astencil,stencil)
    end

    function $f(a::StencilArray,stencil::AbstractVector) 
      $_f(a.array,a.stencil,stencil)
    end

    function $f(a::StencilArray,target::StencilArray)
      $f(a,target.stencil)
    end

    function $f(a::StencilArray,limits::Tuple{Real,Real},dt::Real) 
      lb,ub = limits
      stencil = lb:dt:ub
      $f(a,stencil)
    end
  end
end

function _restrict(
  fine_vals::AbstractArray{T,N},
  fine_stencil::AbstractVector,
  coarse_stencil::AbstractVector,
  ) where {T,N}

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

function _expand(
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
