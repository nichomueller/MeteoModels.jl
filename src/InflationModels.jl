abstract type InflationModel end

get_parameter(i::InflationModel) = @abstractmethod

struct MultInflation <: InflationModel
  ρ::Real 
end

InflationModel(args...) = MultInflation(1.0)

get_parameter(i::MultInflation) = i.ρ

struct NLLInflation <: InflationModel
  taper::TaperModel
  bounds::Tuple{Real,Real}
  tolerance::Real
  ρ::Base.RefValue{<:Real}
end

function NLLInflation(taper::TaperModel;lower=1e-3,upper=10.0,tolerance=1e-1,ρ=-1.0)
  bounds = (lower,upper)
  NLLInflation(taper,bounds,tolerance,Ref(ρ))
end

get_parameter(i::NLLInflation) = i.ρ[]

function reset_parameter!(i::NLLInflation)
  i.ρ[] = -1.0
  return
end

function optimise!(cache,i::NLLInflation,d::SecondMoment,θ::SecondMoment,y::InType)
  _y,_P = cache
  lower,upper = i.bounds
  P = cov(d)
  R = cov(θ)
  λoptprev = i.ρ[]

  copyto!(_y,y)
  
  function fun(λ)
    λ < lower && return Inf
    @. _P = λ*P + R
    F = cholesky!(_P)
    logdet = 2*sum(log,diag(F.L))
    quad = dot(y,ldiv!(F,_y))
    return logdet + quad
  end

  λres = Optim.optimize(fun,lower,upper)
  λopt = Optim.minimizer(λres)
  err = fun(λoptprev) - fun(λopt)
  i.ρ[] = λopt

  return err
end
