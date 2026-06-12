abstract type InflationModel end

get_parameter(i::InflationModel) = @abstractmethod

struct MultInflation <: InflationModel
  ρ::Real 
end

InflationModel(args...) = MultInflation(1.0)

get_parameter(i::MultInflation) = i.ρ

struct NLLInflation <: InflationModel
  bounds::Tuple{Real,Real}
  tolerance::Real
  ρ::Base.RefValue{<:Real}
end

function NLLInflation(;lower=1e-3,upper=10.0,tolerance=1e-1,ρ=-1.0)
  bounds = (lower,upper)
  NLLInflation(bounds,tolerance,Ref(ρ))
end

get_parameter(i::NLLInflation) = i.ρ[]

function reset_parameter!(i::NLLInflation)
  i.ρ[] = -1.0
  return
end

function optimise!(_d::SecondMoment,i::NLLInflation,d::SecondMoment,θ::SecondMoment,y::InType)
  _y = mean(_d)
  _Σ = cov(_d)
  lower,upper = i.bounds
  Σ = cov(d)
  R = cov(θ)
  ρoptprev = i.ρ[]

  copyto!(_y,y)

  function fun(ρ)
    ρ < lower && return Inf
    @. _Σ = ρ*Σ + R
    F = cholesky!(_P)
    logdet = 2*sum(log,diag(F.L))
    quad = dot(y,ldiv!(F,_y))
    return logdet + quad
  end

  ρres = Optim.optimize(fun,lower,upper)
  ρopt = Optim.minimizer(ρres)
  err = fun(ρoptprev) - fun(ρopt)
  i.ρ[] = ρopt

  return err
end
