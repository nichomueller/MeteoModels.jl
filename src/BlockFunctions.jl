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

function evaluate(f::BlockFunction,x::BlockVector)
  xi = x[Block(1)]
  v = evaluate(f.forms[1],xi)
  vals = Vector{typeof(v)}(undef,blocklength(f))
  for i in eachblock(f)
    vals[i] = evaluate(f.forms[i],x[Block(i)])
  end
  return mortar(vals) 
end

(f::BlockFunction)(x...) = evaluate(f,x...)

function evaluate!(cache::BlockVector{<:Number},f::BlockFunction,x::BlockVector)
  for i in eachblock(f)
    cache[Block(i)] = f.forms[i](x[Block(i)])
  end
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