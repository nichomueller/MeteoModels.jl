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

function optimise!(cache::SecondMoment,i::NLLInflation,d::SecondMoment,y::InType,args...)
  _y = mean(cache)
  _Σ = cov(cache)
  Σ = cov(d)
  lower,upper = i.bounds
  ρoptprev = i.ρ[]

  function fun(ρ)
    ρ < lower && return Inf
    copyto!(_y,y)
    _update_cov!(_Σ,Σ,ρ,args...)
    F = cholesky!(_Σ)
    logdet = 2*sum(log,diag(F.L))
    ldiv!(F,_y)
    return logdet + dot(y,_y)
  end

  ρres = Optim.optimize(fun,lower,upper)
  ρopt = Optim.minimizer(ρres)
  err = fun(ρoptprev) - fun(ρopt)
  i.ρ[] = ρopt

  return err
end

function optimise!(cache::SecondMoment,i::NLLInflation,d::Ensemble,y::InType,args...)
  _y = mean(cache)
  _Σ = cov(cache)
  lower,upper = i.bounds
  ρoptprev = i.ρ[]

  A = anomaly(d)
  Σ = similar(_Σ)
  cov_from_anomaly!(Σ,A)

  function fun(ρ)
    ρ < lower && return Inf
    copyto!(_y,y)
    _update_cov!(_Σ,Σ,ρ,args...)
    F = cholesky!(_Σ)
    logdet = 2*sum(log,diag(F.L))
    ldiv!(F,_y)
    return logdet + dot(y,_y)
  end

  ρres = Optim.optimize(fun,lower,upper)
  ρopt = Optim.minimizer(ρres)
  err = fun(ρoptprev) - fun(ρopt)
  i.ρ[] = ρopt

  return err
end

function _update_cov!(_Σ,Σ,ρ)
  copyto!(_Σ,Σ) 
  rmul!(_Σ,ρ)
end

function _update_cov!(_Σ,Σ,ρ,θ::SecondMoment)
  _update_cov!(_Σ,Σ,ρ)
  _Σ .+= cov(θ)
end
