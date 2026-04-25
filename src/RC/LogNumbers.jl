struct LogNumber{N} <: Real
  value::Float64
  LogNumber{N}(x::Real) where N = new(log(N,Float64(x)))
  LogNumber{N}(v::Float64,::Nothing) where N = new(v)
end

const Log = LogNumber{ℯ}
const Log2 = LogNumber{2}
const Log10 = LogNumber{10}

Base.convert(::Type{LogNumber{N}},x::Real) where N = LogNumber{N}(Float64(x),nothing)
Base.convert(::Type{T},x::LogNumber) where T<:Real = T(x.value)
Base.Float64(x::LogNumber) = x.value
Base.float(x::LogNumber) = x

Base.promote_rule(::Type{LogNumber{N}},::Type{S}) where {N,S<:Real} = LogNumber{N}

Base.:+(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value + b.value,nothing)
Base.:-(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value - b.value,nothing)
Base.:*(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value * b.value,nothing)
Base.:/(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value / b.value,nothing)
Base.:-(a::LogNumber{N}) where N = LogNumber{N}(-a.value,nothing)
Base.:(==)(a::LogNumber,b::LogNumber) = a.value == b.value
Base.isless(a::LogNumber,b::LogNumber) = isless(a.value,b.value)

Base.zero(::Type{LogNumber{N}}) where N = LogNumber{N}(0.0,nothing)
Base.one(::Type{LogNumber{N}}) where N = LogNumber{N}(1.0,nothing)
Base.abs(x::LogNumber{N}) where N = LogNumber{N}(abs(x.value),nothing)
Base.sign(x::LogNumber) = sign(x.value)
Base.inv(x::LogNumber{N}) where N = LogNumber{N}(inv(x.value),nothing)
Base.sqrt(x::LogNumber{N}) where N = LogNumber{N}(sqrt(x.value),nothing)
Base.:^(a::LogNumber{N},b::Integer) where N = LogNumber{N}(a.value^b,nothing)
