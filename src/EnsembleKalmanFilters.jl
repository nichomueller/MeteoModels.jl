function KalmanFilter(transition::Model,observation::Model,prior::Ensemble{DoNotUpdateCov})
  obs_prior = change_style(observation(prior))
  cache = KalmanCache(transition,observation,prior)
  KalmanFilter(transition,observation,prior,obs_prior,cache)
end

function KalmanFilter(transition::Function,observation::Function,prior::Ensemble{DoNotUpdateCov})
  k = 1
  transk = transition(k)
  obsk = observation(k)
  obs_prior = change_style(obsk(prior))
  cache = KalmanCache(transk,obsk,prior)
  FunctionKalmanFilter(transition,observation,prior,obs_prior,cache)
end