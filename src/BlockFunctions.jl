struct BlockFunction{F<:AbstractVector} <: Function  
  forms::F 
end

BlockArrays.blocklength(f::BlockFunction) = length(f.forms)
BlockArrays.eachblock(f::BlockFunction) = Base.OneTo(blocklength(f))

jac(f::Function,x::AbstractArray) = jacobian(f,x)

function jac(f::BlockFunction,x::AbstractArray{T}) where T   
  @notimplemented "A block function must receive a block array as input"
end

function jac(f::BlockFunction,x::BlockVector{T}) where T   
  nb = blocklength(f)
  J = Matrix{Matrix{T}}(undef,nb,nb)
  for i in 1:nb 
    J[i,i] = jac(f.forms[i],x[Block(i)])
  end
  fill_nondiag_blocks!(J)
  mortar(J)
end

(f::BlockFunction)(x...) = evaluate(f,x...)

function return_cache(f::BlockFunction,x::Number)
  fill(x,blocklength(f)) 
end

function return_cache(f::Broadcasting{<:BlockFunction},x::BlockVector)
  fi = Broadcasting(f.f.forms[1]) 
  xi = x[Block(1)]
  ci = return_cache(fi,xi)
  vi = evaluate!(ci,fi,xi)
  data = Vector{typeof(vi)}(undef,blocklength(x))
  for i in eachblock(x)
    data[i] = evaluate!(ci,Broadcasting(f.f.forms[i]),x[Block(i)])
  end
  mdata = mortar(data)
  # data = similar(x) 
  return ci,mdata
end

function evaluate!(cache,f::BlockFunction,x::Number)
  for i in eachblock(f)
    cache[i] = f.forms[i](x)
  end
  cache
end 

function evaluate!(cache,f::Broadcasting{<:BlockFunction},x::BlockVector)
  ci,data = cache 
  for i in eachblock(f.f)
    println(i)
    data[Block(i)] = evaluate!(ci,Broadcasting(f.f.forms[i]),x[Block(i)])
  end
  data
end 

# utils 

function fill_nondiag_blocks!(J::AbstractMatrix{<:AbstractMatrix{T}}) where T
  @assert size(J,1) == size(J,2)
  n = size(J,1)
  s = diag_sizes(J)
  for i in 1:n
    for j in i+1:n
      J[i,j] = zeros(T,s[i],s[j])
      J[j,i] = zeros(T,s[j],s[i])
    end
  end
end

function diag_sizes(J::AbstractMatrix{<:AbstractMatrix})
  @assert size(J,1) == size(J,2)
  n = size(J,1)
  s = zeros(Int,n)
  for i in 1:n 
    s[i] = size(J[i,i],1)
  end
  return s 
end