struct NovoaTrainMethod <: TrainMethod
    rho_range::Tuple{Float64,Float64}
    sigin_log_range::Tuple{Float64,Float64}
    tikh_values::Vector{Float64}
    N_folds::Int
    N_val::Int
    noise_level::Float64
    N_grid::Int
    augment_data::Bool
end

function NovoaTrainMethod(;
        rho_range::Tuple{Float64,Float64}       = (0.7, 1.05),
        sigin_log_range::Tuple{Float64,Float64} = (log10(1e-5), log10(1.0)),
        tikh_values::Vector{Float64}            = [1e-16, 1e-12, 1e-10, 1e-8],
        N_folds::Int         = 4,
        N_val::Int           = 200,
        noise_level::Float64 = 0.03,
        N_grid::Int          = 4,
        augment_data::Bool   = true,
    )
    NovoaTrainMethod(rho_range, sigin_log_range, tikh_values,
                    N_folds, N_val, noise_level, N_grid, augment_data)
end

# ——————————————————————————————————————————————————————————
# Core ESN primitives (pure functions, no struct mutation)
# ——————————————————————————————————————————————————————————

function _nova_step(r_in, u, W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
    u_aug   = vcat(u ./ norm_vec, bias_in)
    r_new   = tanh.(Win * (sigma_in .* u_aug) .+ rho .* (W * r_in))
    vcat(r_new, bias_out), r_new
end

function _nova_open_loop(r0, U, W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
    N_units = length(r0)
    N_bout  = length(bias_out)
    Nt      = size(U, 2)
    Xa      = zeros(N_units + N_bout, Nt + 1)
    Xa[1:N_units, 1]    .= r0
    Xa[N_units+1:end, 1] .= bias_out
    r = copy(r0)
    for i in 1:Nt
        xa, r = _nova_step(r, view(U, :, i), W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
        Xa[:, i+1] .= xa
    end
    Xa
end

function _nova_closed_loop(xa0, N, Wout, W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
    N_units = size(Win, 1)
    N_dim   = length(norm_vec)
    Y       = zeros(N_dim, N + 1)
    y       = Wout' * xa0
    Y[:, 1] .= y
    r = copy(xa0[1:N_units])
    for i in 1:N
        xa, r = _nova_step(r, y, W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
        y     = Wout' * xa
        Y[:, i+1] .= y
    end
    Y
end

# ——————————————————————————————————————————————————————————
# Data helpers
# ——————————————————————————————————————————————————————————

_ensure3d(U::AbstractMatrix)          = reshape(U, size(U, 1), size(U, 2), 1)
_ensure3d(U::AbstractArray{<:Any,3}) = U

"""
Augment training data by appending two scaled copies along the trajectory axis (dim 3).
Matches `main_training.py`: U = [U, 0.1*U, -0.01*U].
"""
function _augment3d(U::AbstractArray{<:Number,3})
    cat(U, 0.1 .* U, -0.01 .* U; dims=3)
end

"""
Add noise to training INPUTS (not targets), with a fixed RNG seed.
Matches `main_training.py`: noise is pre-computed once before any search.
Loop order (dim first, then trajectory) matches Python's `for j ... for i ...`.
"""
function _add_input_noise(U_train::AbstractArray{<:Number,3}, noise_level::Float64)
    N_dim, N_train, N_traj = size(U_train)
    U_noisy = copy(U_train)
    rng = MersenneTwister(0)
    for j in 1:N_dim
        for kk in 1:N_traj
            σ = std(view(U_train, j, :, kk)) * noise_level
            U_noisy[j, :, kk] .+= randn(rng, N_train) .* σ
        end
    end
    U_noisy
end

# ——————————————————————————————————————————————————————————
# RVC-Noise validation for one (rho, sigma_in) pair
# ——————————————————————————————————————————————————————————

function _rvc_noise(
        rho::Float64, log10_sin::Float64,
        tikh_values::AbstractVector{Float64},
        U_wash_arr::AbstractArray,    # (N_dim × N_wash  × N_traj)
        U_train_noisy::AbstractArray, # (N_dim × N_train × N_traj), noise pre-applied to inputs
        Y_train_arr::AbstractArray,   # (N_dim × N_train × N_traj), clean one-step-ahead targets
        norm_vec::AbstractVector,
        W, Win, bias_in, bias_out,
        N_folds::Int, N_val::Int,
    )

    sigma_in               = 10.0^log10_sin
    N_dim, N_train, N_traj = size(U_train_noisy)
    N_units                = size(W, 1)
    N_aug                  = N_units + length(bias_out)
    n_tikh                 = length(tikh_values)
    norm_sq                = norm_vec .^ 2

    Xa_all = Vector{Matrix{Float64}}(undef, N_traj)
    LHS    = zeros(N_aug, N_aug)
    RHS    = zeros(N_aug, N_dim)

    for kk in 1:N_traj
        Xa_w  = _nova_open_loop(zeros(N_units), view(U_wash_arr, :, :, kk),
                                W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
        r_end = Xa_w[1:N_units, end]

        # Drive ESN with pre-noised inputs
        Xa         = _nova_open_loop(r_end, view(U_train_noisy, :, :, kk),
                                     W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
        Xa_all[kk] = Xa

        # Accumulate normal equations against clean targets
        X_reg = Xa[:, 2:end]
        LHS  .+= X_reg * X_reg'
        RHS  .+= X_reg * view(Y_train_arr, :, :, kk)'
    end

    LHS_work = copy(LHS)
    Wout_all = Vector{Matrix{Float64}}(undef, n_tikh)
    for j in 1:n_tikh
        δ = j == 1 ? tikh_values[1] : tikh_values[j] - tikh_values[j-1]
        @views LHS_work[diagind(LHS_work)] .+= δ
        Wout_all[j] = Matrix(LHS_work \ RHS)
    end

    N_fw = max(1, (N_train - N_val) ÷ max(N_folds - 1, 1))
    Mean = zeros(n_tikh)

    for kk in 1:N_traj
        Xa    = Xa_all[kk]
        U_ref = view(U_train_noisy, :, :, kk)   # closed-loop target: noisy inputs (matches Python)
        for i in 1:N_folds
            p = (i - 1) * N_fw
            if p + N_val > N_train
                break
            end
            Y_val = U_ref[:, p+1:p+N_val]
            for j in 1:n_tikh
                Ypred = _nova_closed_loop(Xa[:, p+1], N_val - 1, Wout_all[j],
                                          W, Win, norm_vec, bias_in, bias_out, sigma_in, rho)
                mse   = mean((Y_val .- Ypred) .^ 2) / mean(norm_sq)
                val   = log10(max(mse, 1e-30))
                Mean[j] += isfinite(val) ? val : 10.0 * N_folds * N_traj
            end
        end
    end

    best_j    = argmin(Mean)
    best_mse  = Mean[best_j] / (N_folds * N_traj)
    best_tikh = tikh_values[best_j]
    best_Wout = Wout_all[best_j]

    best_mse, best_tikh, best_Wout
end

# ——————————————————————————————————————————————————————————
# TrainMethod interface
# ——————————————————————————————————————————————————————————

function train(m::NovoaTrainMethod, a::NovoaESN,
               U_wash::AbstractArray, U_train::AbstractArray, Y_train::AbstractArray)
    train!(nothing, m, a, U_wash, U_train, Y_train)
end

function train_cache(::NovoaTrainMethod, ::NovoaESN, args...)
    nothing
end

function train!(::Nothing, m::NovoaTrainMethod, a::NovoaESN,
                U_wash::AbstractArray, U_train::AbstractArray, Y_train::AbstractArray)

    U_wash_arr  = _ensure3d(U_wash)
    U_train_arr = _ensure3d(U_train)
    Y_train_arr = _ensure3d(Y_train)

    # ── Data augmentation: append scaled copies along trajectory axis ─────────
    if m.augment_data
        U_wash_arr  = _augment3d(U_wash_arr)
        U_train_arr = _augment3d(U_train_arr)
        Y_train_arr = _augment3d(Y_train_arr)
    end

    N_dim, _, N_traj = size(U_train_arr)
    @assert N_dim == a.N_dim "U_train first dim $N_dim ≠ ESN N_dim $(a.N_dim)"
    @assert size(Y_train_arr) == size(U_train_arr) "U_train and Y_train must have the same shape"

    # ── Normalization from (augmented) training data ──────────────────────────
    lo = fill( Inf, N_dim)
    hi = fill(-Inf, N_dim)
    for kk in 1:N_traj
        lo = min.(lo, vec(minimum(view(U_train_arr, :, :, kk), dims=2)))
        hi = max.(hi, vec(maximum(view(U_train_arr, :, :, kk), dims=2)))
    end
    nrm = hi .- lo
    nrm[nrm .== 0.0] .= 1.0
    a.norm_vec .= nrm

    # ── Pre-compute noisy inputs once (fixed seed, noise on inputs) ───────────
    U_train_noisy = _add_input_noise(U_train_arr, m.noise_level)

    # ── Coarse grid search ────────────────────────────────────────────────────
    rho_vals = range(m.rho_range[1],       m.rho_range[2];       length=m.N_grid)
    sin_vals = range(m.sigin_log_range[1], m.sigin_log_range[2]; length=m.N_grid)

    best_mse  = Inf
    best_rho  = Float64(rho_vals[1])
    best_sin  = Float64(sin_vals[1])
    best_tikh = m.tikh_values[1]

    for rho in rho_vals, log10_sin in sin_vals
        mse, tikh, _ = _rvc_noise(
            Float64(rho), Float64(log10_sin), m.tikh_values,
            U_wash_arr, U_train_noisy, Y_train_arr,
            a.norm_vec, a.W, a.Win, a.bias_in, a.bias_out,
            m.N_folds, m.N_val,
        )
        if mse < best_mse
            best_mse  = mse
            best_rho  = Float64(rho)
            best_sin  = Float64(log10_sin)
            best_tikh = tikh
        end
    end

    # ── Nelder-Mead local refinement ─────────────────────────────────────────
    optfun = x -> begin
        mse, _, _ = _rvc_noise(
            x[1], x[2], m.tikh_values,
            U_wash_arr, U_train_noisy, Y_train_arr,
            a.norm_vec, a.W, a.Win, a.bias_in, a.bias_out,
            m.N_folds, m.N_val,
        )
        mse
    end
    result = optimize(optfun, [best_rho, best_sin], NelderMead(),
                      Optim.Options(iterations=8, show_trace=false))
    if Optim.minimum(result) < best_mse
        best_rho = Optim.minimizer(result)[1]
        best_sin = Optim.minimizer(result)[2]
        best_mse, best_tikh, _ = _rvc_noise(
            best_rho, best_sin, m.tikh_values,
            U_wash_arr, U_train_noisy, Y_train_arr,
            a.norm_vec, a.W, a.Win, a.bias_in, a.bias_out,
            m.N_folds, m.N_val,
        )
    end

    # ── Final Wout: noisy inputs + clean targets + best tikh ─────────────────
    sigma_in_best = 10.0^best_sin
    N_aug = a.N_units + length(a.bias_out)
    LHS   = zeros(N_aug, N_aug)
    RHS   = zeros(N_aug, N_dim)
    for kk in 1:N_traj
        Xa_w  = _nova_open_loop(zeros(a.N_units), view(U_wash_arr, :, :, kk),
                                a.W, a.Win, a.norm_vec, a.bias_in, a.bias_out,
                                sigma_in_best, best_rho)
        r_end = Xa_w[1:a.N_units, end]
        Xa    = _nova_open_loop(r_end, view(U_train_noisy, :, :, kk),
                                a.W, a.Win, a.norm_vec, a.bias_in, a.bias_out,
                                sigma_in_best, best_rho)
        X_reg = Xa[:, 2:end]
        LHS  .+= X_reg * X_reg'
        RHS  .+= X_reg * view(Y_train_arr, :, :, kk)'
    end
    LHS[diagind(LHS)] .+= best_tikh
    final_Wout = Matrix(LHS \ RHS)

    # ── Update ESN ────────────────────────────────────────────────────────────
    a.sigma_in = sigma_in_best
    a.rho      = best_rho
    copyto!(a.Wout, final_Wout)
    fill!(a.state, 0.0)
    a.initialised = false

    # ── Return open-loop states for first trajectory (diagnostic) ─────────────
    Xa_w1 = _nova_open_loop(zeros(a.N_units), view(U_wash_arr, :, :, 1),
                             a.W, a.Win, a.norm_vec, a.bias_in, a.bias_out,
                             a.sigma_in, a.rho)
    Xa1   = _nova_open_loop(Xa_w1[1:a.N_units, end], view(U_train_noisy, :, :, 1),
                             a.W, a.Win, a.norm_vec, a.bias_in, a.bias_out,
                             a.sigma_in, a.rho)
    Xa1[:, 2:end]
end
