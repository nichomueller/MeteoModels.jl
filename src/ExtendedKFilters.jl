function linearize(f::KalmanFilter,x)
  tx = linearize(f.transition,x)
  ox = linearize(f.observation,x)
  KalmanFilter(tx,ox,f.prior,f.cache)
end

function linearize_transition(f::KalmanFilter,x)
  tx = linearize(f.transition,x)
  KalmanFilter(tx,f.observation,f.prior,f.cache)
end

function linearize_observation(f::KalmanFilter,x)
  ox = linearize(f.observation,x)
  KalmanFilter(f.transition,ox,f.prior,f.cache)
end

const ExtendedKalmanFilter{A<:StochasticLinearizedModel,B<:StochasticLinearizedModel,C<:SecondMoment} = KalmanFilter{A,B,C}

function predict!(posterior::SecondMoment,f::ExtendedKalmanFilter,y::InType)
  flin = linearize_transition(f,get_prior(f))
  predict!(posterior,flin,y)
  return posterior
end

function update!(posterior::SecondMoment,f::ExtendedKalmanFilter,y::InType)
  flin = linearize_observation(f,posterior)
  update!(posterior,flin,y)
  return posterior
end
