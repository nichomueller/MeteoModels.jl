struct NovoaParamBounds
    n_params::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
end

mutable struct NovoaBiasAwareFilter{A<:BiasAwareKalmanFilter} <: Filter
    filter::A
    num_DA_blind::Int
    inflation::Float64
    param_bounds::Union{Nothing,NovoaParamBounds}
    step::Int
end

function NovoaBiasAwareFilter(
        transition::Model,
        observation::Model,
        prior::Ensemble,
        obs_prior::Ensemble,
        bias_model::RecurrentNeuralNetwork;
        γ::Real                              = 10,
        num_DA_blind::Int                    = 0,
        inflation::Float64                   = 0.0,
        param_bounds::Union{Nothing,NovoaParamBounds} = nothing,
        kwargs...
    )
    bakf = BiasAwareKalmanFilter(
        transition, observation, prior, obs_prior, bias_model; γ, kwargs...
    )
    NovoaBiasAwareFilter(bakf, num_DA_blind, inflation, param_bounds, 0)
end

get_prior(f::NovoaBiasAwareFilter)             = get_prior(f.filter)
get_observation_prior(f::NovoaBiasAwareFilter) = get_observation_prior(f.filter)
get_transition_model(f::NovoaBiasAwareFilter)  = get_transition_model(f.filter)
get_observation_model(f::NovoaBiasAwareFilter) = get_observation_model(f.filter)
get_noise(f::NovoaBiasAwareFilter)             = get_noise(f.filter)
get_observation_noise(f::NovoaBiasAwareFilter) = get_observation_noise(f.filter)

function reset!(f::NovoaBiasAwareFilter)
    reset!(f.filter)
    f.step = 0
end

function forecast!(posterior::Ensemble, f::NovoaBiasAwareFilter)
    transition!(posterior, f.filter)
end

function _nova_inflate!(posterior::Ensemble, δ::Float64)
    δ == 0.0 && return
    A = anomaly(posterior)
    μ = mean(posterior)
    X = get_ensemble(posterior)
    rmul!(A, 1.0 + δ)
    @inbounds @views for i in 1:ensemble_size(posterior)
        X[:, i] .= μ .+ A[:, i]
    end
end

function _nova_inflate_from!(posterior::Ensemble, Af::AbstractMatrix, δ::Float64)
    copyto!(get_ensemble(posterior), Af)
    update_mean!(posterior)
    update_anomaly!(posterior)
    _nova_inflate!(posterior, δ)
    posterior
end

function _freeze_params!(posterior::Ensemble, Af::AbstractMatrix, pb::NovoaParamBounds)
    X = get_ensemble(posterior)
    μ = mean(posterior)
    A = anomaly(posterior)
    n = size(X, 1)
    rows = (n - pb.n_params + 1):n
    @views X[rows, :] .= Af[rows, :]
    @views μ[rows]    .= vec(mean(Af[rows, :], dims=2))
    @inbounds @views for i in axes(X, 2)
        A[rows, i] .= X[rows, i] .- μ[rows]
    end
    posterior
end

function _params_physical(posterior::Ensemble, pb::NovoaParamBounds)
    X = get_ensemble(posterior)
    n = size(X, 1)
    rows = (n - pb.n_params + 1):n
    @inbounds for col in axes(X, 2)
        for (i, row) in enumerate(rows)
            if X[row, col] < pb.lo[i] || X[row, col] > pb.hi[i]
                return false
            end
        end
    end
    return true
end

function analyse!(posterior::Ensemble, f::NovoaBiasAwareFilter, z::AbstractVector)
    f.step += 1
    bakf = f.filter
    pb   = f.param_bounds

    Af = pb !== nothing ? copy(get_ensemble(posterior)) : nothing

    if f.step <= f.num_DA_blind
        analyse!(posterior, bakf.filter, z)
        observation!(bakf, posterior)
        ỹa = posterior_innovation!(bakf, z)
        evaluate!(bakf.cache.eval_cache, bakf.bias_model, ỹa)
        if pb !== nothing
            _freeze_params!(posterior, Af, pb)
        end
    else
        analyse!(posterior, bakf, z)
        if pb !== nothing && !_params_physical(posterior, pb)
            _nova_inflate_from!(posterior, Af, 0.05)
            if !_params_physical(posterior, pb)
                copyto!(get_ensemble(posterior), Af)
                update_mean!(posterior)
                update_anomaly!(posterior)
            end
        end
    end

    _nova_inflate!(posterior, f.inflation)

    posterior
end

function analyse!(posterior::Ensemble, f::NovoaBiasAwareFilter)
    analyse!(posterior, f.filter)
    posterior
end
