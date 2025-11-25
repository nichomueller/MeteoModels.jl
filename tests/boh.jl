# Helper in-place mean (no allocation of length n vectors)
function mean_into!(μ::AbstractVector, X::AbstractMatrix)
    # μ .= mean(X, dims=2)  # this would allocate a 1-column array; avoid
    @inbounds begin
        fill!(μ, zero(eltype(μ)))
        N = size(X, 2)
        for j in 1:N
            @views μ .+= X[:, j]
        end
        μ ./= N
    end
    return μ
end

# In-place anomalies: Δ = X .- μ (Δ must be preallocated same size as X)
function anomalies_into!(Δ::AbstractMatrix, X::AbstractMatrix, μ::AbstractVector)
    @inbounds for j in 1:size(X,2)
        @views Δ[:, j] .= X[:, j] .- μ
    end
    return Δ
end

# Predict step for a linear model A: Xf <- A * Xf  (safe: use temp to avoid aliasing)
function predict_linear!(
    _c::MeteoModels.EnsembleKalmanCache, 
    _e::KalmanEnsemble, 
    op::MeteoModels.EnsembleKalmanOperators, 
    obs::Observation)
    X = _e.data            # n × Ne
    A = op.op.trans_model # n × n
    n, Ne = size(X)
    tmp = similar(X)      # you might want to preallocate in cache instead
    mul!(tmp, A, X)       # tmp = A * X
    copyto!(X, tmp)       # write result back

    # mean and state covariance into cache.mean / cache.cov
    mean_into!(_c.mean, X)
    anomalies_into!(_c.cov, X, _c.mean) # temporarily reusing _c.cov as δ matrix
    # Now compute state covariance C = δ * δ' / (Ne - 1)
    mul!(_c.cov, _c.cov, transpose(_c.cov)) # _c.cov = δ * δ'
    _c.cov ./= (Ne - 1)                    # C now holds state covariance

    return _e
end

###########################################################

kf = Filter(op,copy(iter))
e = kf.iterables
c = kf.cache

x̂ = MeteoModels.get_state(e)
μ = MeteoModels.get_mean(c)
C = MeteoModels.get_cov(c)

copyto!(x̂,op.op.trans_model*x̂)

mean!(μ,x̂)
MeteoModels.cov!(C,x̂,μ)
@assert C ≈ cov(x̂')

ỹ = c.innovation             
S = c.innovation_cov          
K = c.kalman_gain                       

mul!(ỹ,H,x̂,-1.0,0.0)    

mul!(K,C,H')
mul!(S,H,K)
S .+= R                          

F = cholesky!(S)     
rdiv!(K,F)      

ỹ .+= MeteoModels.get_measurement(x)
mul!(x̂,K,ỹ,1.0,1.0) 

###########################################################

_kf = Filter(op,copy(iter))
_e = _kf.iterables
_c = _kf.cache
      
n, Ne = size(_e.data)
m = size(H, 1)

# Reuse fields in cache or allocate temp views as necessary:
Yf = _c.innovation              # use innovation matrix slot as predicted obs Yf (m × Ne)
mean_y = similar(_c.mean, m)    # temporary mean for observations
Δx = similar(_e.data)                # anomalies (n × Ne) local temp
Δy = similar(Yf)               # anomalies in observation space (m × Ne)

# 1) Predicted observations Yf = H * X
mul!(Yf, H, _e.data)                 # Yf (m × Ne) = H * X

# 2) Means
mean_into!(_c.mean, _e.data)          # _c.mean is n-vector
mean_into!(mean_y, Yf)         # mean of Yf (m-vector)

# 3) Anomalies Δx, Δy
anomalies_into!(Δx, _e.data, _c.mean)    # Δx = X - mean_x
anomalies_into!(Δy, Yf, mean_y)   # Δy = Yf - mean_y

# 4) Covariances (sample covariances)
# P_xy = (Δx * Δy') / (Ne - 1)   -> n × m
P_xy = similar(_c.kalman_gain)          # n × m
mul!(P_xy, Δx, transpose(Δy))
P_xy ./= (Ne - 1)

# P_yy = (Δy * Δy')/(Ne-1) + R      -> m × m
mul!(_c.innovation_cov, Δy, transpose(Δy))
_c.innovation_cov ./= (Ne - 1)
_c.innovation_cov .+= R

# 5) Solve for K = P_xy * inv(P_yy) using Cholesky
F = cholesky!(_c.innovation_cov)   # factorize in-place (S = P_yy)
# Solve S * X = P_xy'  -> X = S^{-1} * P_xy'
tmp = copy(transpose(P_xy))       # m × n  (P_xy')
ldiv!(F, tmp)                     # tmp := S \ tmp
_K = transpose(tmp)                # n × m -> this is P_xy * inv(S)
copyto!(_c.kalman_gain, _K)         # store in cache

# 6) Create perturbed observations (stochastic EnKF)
y_k = MeteoModels.get_measurement(y)        # m-vector
# create Y_obs (m × Ne): each column = y_k + eps_i, eps_i ~ N(0,R)
# reuse Δy as workspace for Y_obs
# @inbounds for j in 1:Ne
#     # draw perturbation vector
#     # simplest: sqrt of diagonal noise; for full covariance you need multivariate Gaussian
#     for i in 1:m
#         Δy[i, j] = y_k[i] + (sqrt(R[i,i]) * randn())  # cheap stochastic perturbation
#     end
# end

# 7) Update ensemble: X_a = X_f + K * (Y_obs - Yf)
# compute innovation matrix: E = Y_obs - Yf  (reusing tmp as workspace)
tmp1 = zeros(no,no) 
@inbounds for j in 1:Ne
    @views tmp1[:, j] .= Δy[:, j] .- Yf[:, j]
end
# Compute K * E -> n × Ne  (reuse Δx as workspace)
mul!(Δx, _c.kalman_gain, tmp1)
# Add back to forecast ensemble in place
@inbounds for j in 1:Ne
    @views X[:, j] .+= Δx[:, j]
end

# 8) Update cache mean and cov for analysis
mean_into!(_c.mean, X)
anomalies_into!(Δx, X, _c.mean)
mul!(_c.cov, Δx, transpose(Δx))
_c.cov ./= (Ne - 1)