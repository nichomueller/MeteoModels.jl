struct LogNumber{N,T<:Real} <: Real
  value::T
  LogNumber{N}(x::T) where {N,T<:Real} = new{N,T}(x)
end

Base.convert(::Type{LogNumber{N,T}},x::Real) where {N,T<:Real} = LogNumber{N,T}(convert(T,x))
Base.convert(::Type{T},x::LogNumber{N,T}) where {N,T<:Real} = x.value
Base.Float64(x::LogNumber) = Float64(x.value)
Base.float(x::LogNumber{N,T}) where {N,T} = LogNumber{N,float(T)}(float(x.value))

Base.promote_rule(::Type{LogNumber{N,T}},::Type{S}) where {N,T<:Real,S<:Real} = LogNumber{N,promote_type(T,S)}

Base.:+(a::LogNumber{N,T},b::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(a.value + b.value)
Base.:-(a::LogNumber{N,T},b::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(a.value - b.value)
Base.:*(a::LogNumber{N,T},b::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(a.value * b.value)
Base.:/(a::LogNumber{N,T},b::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(a.value / b.value)
Base.:-(a::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(-a.value)
Base.:(==)(a::LogNumber,b::LogNumber) = a.value == b.value
Base.isless(a::LogNumber,b::LogNumber) = isless(a.value,b.value)

Base.zero(::Type{LogNumber{N,T}}) where {N,T} = LogNumber{N,T}(zero(T))
Base.one(::Type{LogNumber{N,T}}) where {N,T} = LogNumber{N,T}(one(T))
Base.abs(x::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(abs(x.value))
Base.sign(x::LogNumber) = sign(x.value)
Base.inv(x::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(inv(x.value))
Base.sqrt(x::LogNumber{N,T}) where {N,T} = LogNumber{N,T}(sqrt(x.value))
Base.:^(a::LogNumber{N,T},b::Integer) where {N,T} = LogNumber{N,T}(a.value^b)
