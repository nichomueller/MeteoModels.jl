mutable struct ConvLog{T<:Real}
  name::String
  maxiter::Int
  atol::T
  verbose::Bool
  num_iters::Int
  values::Vector{T}
end

function ConvLog(name::String,maxiter::Integer,tol::Real,verbose::Bool=true)
  T = typeof(float(tol))
  ConvLog{T}(name,Int(maxiter),T(tol),verbose,0,zeros(T,Int(maxiter)+1))
end

function reset!(log::ConvLog)
  log.num_iters = 0
  fill!(log.values,zero(eltype(log.values)))
  return log
end

converged(log::ConvLog,v::Real) = v <= log.atol
finished(log::ConvLog,v::Real) = (log.num_iters >= log.maxiter) || converged(log,v)

function init!(log::ConvLog,v::Real)
  reset!(log)
  vT = convert(eltype(log.values),v)
  log.values[1] = vT
  if log.verbose
    println("Starting ",log.name," (atol=",log.atol,", maxiter=",log.maxiter,")")
    println("iter=0, value=",vT)
  end
  return finished(log,vT)
end

function update!(log::ConvLog,v::Real)
  log.num_iters += 1
  vT = convert(eltype(log.values),v)
  log.values[log.num_iters+1] = vT
  if log.verbose
    println("iter=",log.num_iters,", value=",vT)
  end
  return finished(log,vT)
end