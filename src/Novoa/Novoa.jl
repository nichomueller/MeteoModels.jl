# Novoa — bias-aware EnKF implementation following Novoa et al. (rBA-EnKF),
# built on MeteoModels data structures.
#
# The rBA-EnKF analysis formula is already implemented in MeteoModels as
# BiasAwareKalmanFilter / BiasAwareEnsembleKalmanFilter.  This module supplies
# the pieces that differ from the default MeteoModels workflow:
#
#   NovoaESN          — ESN with runtime sigma_in/rho, absolute normalization,
#                      vector biases, no leaky integration (NovoaESN.jl).
#
#   NovoaTrainMethod  — Hyperparameter search via grid + Nelder-Mead over
#                      (rho, sigma_in) with RVC-Noise validation (NovoaESNTraining.jl).
#
#   EnSRKFFilter     — Ensemble Square-Root Kalman Filter: deterministic,
#                      anomaly updated via SVD square-root formula (NovoaEnSRKF.jl).
#
#   NovoaBiasAwareFilter — Full rBA-EnKF workflow: blind DA period, multiplicative
#                         inflation, bias-aware analysis (NovoaBiasAwareFilter.jl).
#
# Usage sketch (full workflow):
#
#   # 1. Build and train the bias ESN
#   esn = NovoaESN(N_dim, N_units; connect=5)
#   train(NovoaTrainMethod(), esn, bias_training_data)   # (N_dim × Nt)
#   washout!(esn, washout_data)                         # (N_dim × N_wash)
#
#   # 2. Build prior (DEnKF or EnKF recommended for use with BiasAwareKalmanFilter)
#   prior     = Ensemble(DEnKFStrategy(), X0)
#   obs_prior = observation(prior)
#
#   # 3. Build the full filter
#   f = NovoaBiasAwareFilter(
#       transition, observation, prior, obs_prior, esn;
#       γ = 10, num_DA_blind = 50, inflation = 0.05,
#   )
#
#   # 4. Run the DA loop (standard MeteoModels interface)
#   history = loop(f, observations)
#
# For standalone square-root filtering (without bias correction):
#   prior = Ensemble(EnSRKFStrategy(), X0)
#   f     = KalmanFilter(transition, observation, prior)
#   history = loop(f, observations)

include("NovoaESN.jl")
include("NovoaESNTraining.jl")
# include("NovoaEnSRKF.jl")
include("NovoaBiasAwareFilter.jl")

export NovoaESN
export NovoaTrainMethod
export washout!
# export EnSRKFStrategy
export NovoaBiasAwareFilter
export ParamBounds
