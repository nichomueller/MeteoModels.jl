"""
    struct LogNumber{N} <: Real

A real number whose `value` field is interpreted as a logarithm in base `N`.

Arithmetic operations on two `LogNumber{N}` values act on the stored exponents directly,
so `a * b` adds exponents, `a / b` subtracts them, etc.  This lets hyper-parameter grids
be expressed in log-space while interoperating with plain `Real` arithmetic.

Aliases: `Log = LogNumber{ℯ}`, `Log2 = LogNumber{2}`, `Log10 = LogNumber{10}`.
"""
struct LogNumber{N} <: Real
  value::Float64
  LogNumber{N}(x::Float64) where N = new{N}(x)
end

const Log = LogNumber{ℯ}
const Log2 = LogNumber{2}
const Log10 = LogNumber{10}

Base.convert(::Type{LogNumber{N}},x::Real) where N = LogNumber{N}(Float64(x))
Base.convert(::Type{T},x::LogNumber) where T<:Real = T(x.value)
Base.convert(::Type{LogNumber{N}},x::LogNumber) where N = LogNumber{N}(x.value)
Base.Float64(x::LogNumber) = x.value
Base.float(x::LogNumber) = x

Base.promote_rule(::Type{LogNumber{N}},::Type{S}) where {N,S<:Real} = LogNumber{N}

Base.:+(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value + b.value)
Base.:-(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value - b.value)
Base.:*(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value * b.value)
Base.:/(a::LogNumber{N},b::LogNumber{N}) where N = LogNumber{N}(a.value / b.value)
Base.:-(a::LogNumber{N}) where N = LogNumber{N}(-a.value)
Base.:(==)(a::LogNumber,b::LogNumber) = a.value == b.value
Base.isless(a::LogNumber,b::LogNumber) = isless(a.value,b.value)

Base.zero(::Type{LogNumber{N}}) where N = LogNumber{N}(0.0)
Base.one(::Type{LogNumber{N}}) where N = LogNumber{N}(1.0)
Base.abs(x::LogNumber{N}) where N = LogNumber{N}(abs(x.value))
Base.sign(x::LogNumber) = sign(x.value)
Base.inv(x::LogNumber{N}) where N = LogNumber{N}(inv(x.value))
Base.sqrt(x::LogNumber{N}) where N = LogNumber{N}(sqrt(x.value))
Base.:^(a::LogNumber{N},b::Integer) where N = LogNumber{N}(a.value^b)
