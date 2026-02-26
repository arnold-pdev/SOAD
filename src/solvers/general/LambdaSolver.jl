# LambdaSolver.jl — Three-player (u, λ, θ) inertial dynamics
#
# Integrates the λ-augmented Lagrangian dynamics for general PDE-constrained
# optimization.  All three players evolve simultaneously; no nested solves.
#
# The Lagrangian is
#
#   L(u, θ, λ) = J(u, θ) + ⟨λ, R(u, θ)⟩
#
# where R(u, θ) = A(θ)u − f is the equilibrium residual.
#
# Stationarity conditions (saddle point):
#   D_u L = 0  →  ∂J/∂u + A(θ)ᵀλ = 0       (adjoint equation)
#   D_λ L = 0  →  R(u, θ) = 0               (state equation)
#   D_θ L = 0  →  ∂J/∂θ + (∂R/∂θ)ᵀλ = 0   (design optimality)
#
# Three-player inertial dynamics (forces to be supplied by the user via the
# problem interface — see Problem.jl for F_u, F_λ, F_θ conventions):
#
#   m_u · ü  +  c_u(t) · u̇  =  F_u      [u player]
#   m_λ · λ̈  +  c_λ(t) · λ̇  =  F_λ      [λ player]
#   m_θ · θ̈  +  c_θ(t) · θ̇  =  F_θ      [θ player]
#
# Multirate timestepping
# ----------------------
# The three players may operate at different effective timescales.  Two modes:
#
# Fixed ratio mode  (n_inner_ul, n_inner_u specified as integers):
#   Each outer step of θ is preceded by n_inner_ul steps of the (u, λ) pair,
#   each of which is preceded by n_inner_u steps of u alone relative to λ.
#   n_inner_ul = 1, n_inner_u = 1 recovers simultaneous single-step updates.
#
# Adaptive mode  (inner_tol_ul, inner_tol_u specified as Float64):
#   Before each θ step, (u, λ) inner steps are taken until
#       ‖F_u‖ / ‖F_θ‖ < inner_tol_ul   AND   ‖F_λ‖ / ‖F_θ‖ < inner_tol_ul
#   Before each λ step (within the inner loop), u steps are taken until
#       ‖F_u‖ / ‖F_λ‖ < inner_tol_u
#   A maximum of max_inner steps caps the loop to prevent infinite cycling.
#
# The two modes are selected per ratio independently: pass an Int for fixed,
# a Float64 for adaptive.  Mixed mode (fixed u/λ, adaptive ul) is allowed.
#
# Compliance verification
# -----------------------
# For compliance J = ⟨f, u⟩ the exact adjoint is λ* = −u*.  The solver
# records ‖λ + u‖ / ‖u‖ as `lambda_state_residual`.
#
# FIRESchedule + RK4 incompatibility is enforced at construction time.

import LinearAlgebra: norm, dot
import Printf: @printf

# ============================================================
#  Multirate policy types
# ============================================================

"""
    FixedRatio(n::Int)

Take exactly `n` inner steps of the fast player(s) per outer step.
"""
struct FixedRatio
    n ::Int
end

"""
    AdaptiveTol(tol::Float64, max_steps::Int)

Take inner steps until the force-ratio falls below `tol`, capped at
`max_steps` to prevent infinite cycling.
"""
struct AdaptiveTol
    tol       ::Float64
    max_steps ::Int
end

# ============================================================
#  Struct
# ============================================================

"""
    LambdaSolver

Three-player inertial solver for the λ-augmented Lagrangian dynamics.

Constructor
-----------
```julia
s = LambdaSolver(
    problem,
    state_schedule,
    lambda_schedule,
    design_schedule,
    integrator;              # SymplecticEuler() or RK4Integrator()
    dt            = nothing,
    T             = 10.0,
    stride        = 10,
    save_fields   = false,
    lambda_init   = nothing, # nothing → zeros; :neg_state → −u₀; or Vector
    # Multirate: u vs λ  (how many u-steps per λ-step)
    inner_u       = FixedRatio(1),   # Int or Float64 → FixedRatio / AdaptiveTol
    # Multirate: (u,λ) vs θ  (how many (u,λ)-steps per θ-step)
    inner_ul      = FixedRatio(1),
    max_inner     = 1000,    # cap for adaptive loops
    verbose       = false
)
```

Multirate shorthands: pass an `Int` for `FixedRatio(n)`, a `Float64` for
`AdaptiveTol(tol, max_inner)`.

Fields
------
- `state`   : current u (length state_dim(problem))
- `lambda`  : current λ (same length as state)
- `design`  : current θ (length design_dim(problem))
- `vel_*`   : velocities for each player

Diagnostics (in SolverHistory)
-------------------------------
- `grad_lambda_norm`      : ‖F_λ‖
- `vel_lambda_norm`       : ‖λ̇‖
- `lambda_state_residual` : ‖λ + u‖ / ‖u‖  (→ 0 for compliance at saddle)
"""
mutable struct LambdaSolver{P, IS, LS, DS, I, MUL, MU} <: AbstractSolver
    problem        ::P

    state_sched    ::IS
    lambda_sched   ::LS
    design_sched   ::DS

    integrator     ::I

    # Current iterate
    state          ::Vector{Float64}
    lambda         ::Vector{Float64}
    design         ::Vector{Float64}

    vel_state      ::Vector{Float64}
    vel_lambda     ::Vector{Float64}
    vel_design     ::Vector{Float64}

    # Multirate policies
    inner_ul       ::MUL   # (u,λ) steps per θ step
    inner_u        ::MU    # u steps per λ step

    dt             ::Float64
    T              ::Float64
    max_inner      ::Int
    stride         ::Int
    save_fields    ::Bool
    verbose        ::Bool
end

# ---- constructor helpers ---------------------------------------------------

_to_policy(x::Int,     max_inner) = FixedRatio(x)
_to_policy(x::Float64, max_inner) = AdaptiveTol(x, max_inner)
_to_policy(x::FixedRatio,  _)     = x
_to_policy(x::AdaptiveTol, _)     = x

function LambdaSolver(
    problem,
    state_sched,
    lambda_sched,
    design_sched,
    integrator = SymplecticEuler();
    dt          = nothing,
    T           = 10.0,
    stride      = 10,
    save_fields = false,
    lambda_init = nothing,
    inner_ul    = FixedRatio(1),
    inner_u     = FixedRatio(1),
    max_inner   = 1000,
    verbose     = false
)
    # FIRE + RK4 incompatibility check
    if integrator isa RK4Integrator
        for (sched, name) in ((state_sched,  "state_sched"),
                              (lambda_sched, "lambda_sched"),
                              (design_sched, "design_sched"))
            if hasproperty(sched, :α_current)
                error("LambdaSolver: RK4Integrator is not compatible with " *
                      "FIRESchedule (passed as $name). Use SymplecticEuler.")
            end
        end
    end

    state0  = initial_state(problem)
    design0 = initial_design(problem)
    enforce_state_bcs!(problem, state0)
    enforce_design_bcs!(problem, design0)

    lambda0 = if lambda_init === nothing
        zeros(length(state0))
    elseif lambda_init === :neg_state
        -copy(state0)
    else
        convert(Vector{Float64}, lambda_init)
    end

    dt_actual  = (dt === nothing) ? stable_dt(problem, state0, design0) : Float64(dt)
    pol_ul     = _to_policy(inner_ul, max_inner)
    pol_u      = _to_policy(inner_u,  max_inner)

    LambdaSolver(
        problem,
        state_sched, lambda_sched, design_sched,
        integrator,
        state0, lambda0, design0,
        zeros(length(state0)), zeros(length(lambda0)), zeros(length(design0)),
        pol_ul, pol_u,
        dt_actual, Float64(T), max_inner, stride, save_fields, verbose
    )
end

current_state(s::LambdaSolver)  = s.state
current_design(s::LambdaSolver) = s.design
current_lambda(s::LambdaSolver) = s.lambda

_is_fire(sched) = hasproperty(sched, :α_current)

# ============================================================
#  Force functions
# ============================================================
#
# NOTE: The expressions for F_u, F_λ, F_θ below are PLACEHOLDERS.
# They encode the current (incorrect) form and are marked clearly so
# the user can substitute the correct variational expressions derived
# from the Lagrangian.  Only these three blocks need to change.

function _forces(prob, state, lambda, design)
    # ---- quantities used by multiple forces ----
    dJdu   = obj_state_grad(prob, state, design)
    Aλ     = apply_stiffness(prob, design, lambda)
    _, dJdθ = energy_gradients(prob, state, design)
    dRdθ_λ = residual_design_grad(prob, state, design, lambda)

    # ---- F_u : force on state u ----
    # PLACEHOLDER — replace with correct D_u L expression
    F_u = @. -dJdu - Aλ

    # ---- F_λ : force on adjoint λ ----
    # PLACEHOLDER — replace with correct D_λ L expression
    F_λ = @. -dJdu - Aλ

    # ---- F_θ : force on design θ ----
    # This form (−∂J/∂θ − (∂R/∂θ)ᵀλ) is the correct full design sensitivity.
    F_θ = @. -dJdθ - dRdθ_λ

    return F_u, F_λ, F_θ
end

# Force closures for RK4 substages: each takes the varying player's
# current position x and time τ, with the other two players frozen.
function _F_u_fn(prob, design_now, lambda_now)
    (x, τ) -> begin
        dj = obj_state_grad(prob, x, design_now)
        al = apply_stiffness(prob, design_now, lambda_now)
        @. -dj - al   # PLACEHOLDER — match F_u above
    end
end

function _F_λ_fn(prob, state_now, design_now)
    (x, τ) -> begin
        dj = obj_state_grad(prob, state_now, design_now)
        al = apply_stiffness(prob, design_now, x)
        @. -dj - al   # PLACEHOLDER — match F_λ above
    end
end

function _F_θ_fn(prob, state_now, lambda_now)
    (x, τ) -> begin
        _, djdx = energy_gradients(prob, state_now, x)
        dr      = residual_design_grad(prob, state_now, x, lambda_now)
        @. -djdx - dr
    end
end

# ============================================================
#  Inner loop helpers
# ============================================================

# Single u step (λ and θ frozen)
function _step_u!(s, k, t, F_u, is_fire_state)
    sp = step!(s.state_sched, k, t, F_u, s.vel_state)
    fu_fn = _F_u_fn(s.problem, s.design, s.lambda)
    integrate!(s.integrator, s.state, s.vel_state, fu_fn, sp, s.dt, t;
               is_fire=is_fire_state)
    enforce_state_bcs!(s.problem, s.state)
end

# Single λ step (u and θ frozen)
function _step_λ!(s, k, t, F_λ, is_fire_lambda)
    sp = step!(s.lambda_sched, k, t, F_λ, s.vel_lambda)
    fλ_fn = _F_λ_fn(s.problem, s.state, s.design)
    integrate!(s.integrator, s.lambda, s.vel_lambda, fλ_fn, sp, s.dt, t;
               is_fire=is_fire_lambda)
    enforce_state_bcs!(s.problem, s.lambda)
end

# Single θ step (u and λ frozen)
function _step_θ!(s, k, t, F_θ, is_fire_design)
    sp = step!(s.design_sched, k, t, F_θ, s.vel_design)
    fθ_fn = _F_θ_fn(s.problem, s.state, s.lambda)
    integrate!(s.integrator, s.design, s.vel_design, fθ_fn, sp, s.dt, t;
               is_fire=is_fire_design)
    enforce_design_bcs!(s.problem, s.design)
end

# Number of u sub-steps to take before a λ step, given policy
function _inner_u_steps(policy::FixedRatio, F_u, F_λ, h)
    return policy.n
end

function _inner_u_steps(policy::AdaptiveTol, F_u, F_λ, h)
    norm_Fλ = _l2norm(F_λ, h)
    norm_Fu = _l2norm(F_u, h)
    # If F_λ is negligible, no sub-steps needed
    norm_Fλ < eps() && return 0
    norm_Fu / norm_Fλ < policy.tol && return 0
    return policy.max_steps   # caller will break early when tol met
end

# Number of (u,λ) sub-steps to take before a θ step, given policy
function _inner_ul_steps(policy::FixedRatio, F_u, F_λ, F_θ, h)
    return policy.n
end

function _inner_ul_steps(policy::AdaptiveTol, F_u, F_λ, F_θ, h)
    norm_Fθ = _l2norm(F_θ, h)
    norm_Fθ < eps() && return 0
    ratio_u = _l2norm(F_u, h) / norm_Fθ
    ratio_λ = _l2norm(F_λ, h) / norm_Fθ
    (ratio_u < policy.tol && ratio_λ < policy.tol) && return 0
    return policy.max_steps
end

# ============================================================
#  solve!
# ============================================================

"""
    solve!(s::LambdaSolver) -> SolverHistory

Run the three-player dynamics to time T using the configured multirate policy.
"""
function solve!(s::LambdaSolver)
    h_grid  = s.problem.grid.h
    history = SolverHistory()
    n_steps = round(Int, s.T / s.dt)
    k       = 0
    t       = 0.0

    is_fire_state  = _is_fire(s.state_sched)
    is_fire_lambda = _is_fire(s.lambda_sched)
    is_fire_design = _is_fire(s.design_sched)

    for step in 1:n_steps
        k += 1
        t  = step * s.dt

        # ------------------------------------------------------------------
        # Forces at current (u, λ, θ)
        # ------------------------------------------------------------------
        F_u, F_λ, F_θ = _forces(s.problem, s.state, s.lambda, s.design)

        # ------------------------------------------------------------------
        # Multirate: (u, λ) inner loop before θ step
        # ------------------------------------------------------------------
        n_ul = _inner_ul_steps(s.inner_ul, F_u, F_λ, F_θ, h_grid)

        for inner in 1:n_ul
            # Recompute forces for current inner state
            F_u_i, F_λ_i, _ = _forces(s.problem, s.state, s.lambda, s.design)

            # Multirate: u inner loop before λ step
            n_u = _inner_u_steps(s.inner_u, F_u_i, F_λ_i, h_grid)

            for inner_u in 1:n_u
                F_u_ii, _, _ = _forces(s.problem, s.state, s.lambda, s.design)
                _step_u!(s, k, t, F_u_ii, is_fire_state)

                # Early exit for adaptive u policy
                if s.inner_u isa AdaptiveTol
                    F_u_check, F_λ_check, _ = _forces(s.problem, s.state, s.lambda, s.design)
                    norm_Fλ = _l2norm(F_λ_check, h_grid)
                    norm_Fλ < eps() && break
                    _l2norm(F_u_check, h_grid) / norm_Fλ < s.inner_u.tol && break
                end
            end

            # λ step
            F_u_j, F_λ_j, _ = _forces(s.problem, s.state, s.lambda, s.design)
            _step_λ!(s, k, t, F_λ_j, is_fire_lambda)

            # Early exit for adaptive (u,λ) policy
            if s.inner_ul isa AdaptiveTol
                F_u_c, F_λ_c, F_θ_c = _forces(s.problem, s.state, s.lambda, s.design)
                norm_Fθ = _l2norm(F_θ_c, h_grid)
                if norm_Fθ > eps()
                    ratio_u = _l2norm(F_u_c, h_grid) / norm_Fθ
                    ratio_λ = _l2norm(F_λ_c, h_grid) / norm_Fθ
                    (ratio_u < s.inner_ul.tol && ratio_λ < s.inner_ul.tol) && break
                end
            end
        end

        # ------------------------------------------------------------------
        # θ step
        # ------------------------------------------------------------------
        F_u_f, F_λ_f, F_θ_f = _forces(s.problem, s.state, s.lambda, s.design)
        _step_θ!(s, k, t, F_θ_f, is_fire_design)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        if step % s.stride == 0
            F_u_d, F_λ_d, F_θ_d = _forces(s.problem, s.state, s.lambda, s.design)
            J   = objective(s.problem, s.state, s.design)
            gs  = _l2norm(F_u_d,        h_grid)
            gl  = _l2norm(F_λ_d,        h_grid)
            gd  = _l2norm(F_θ_d,        h_grid)
            vs  = _l2norm(s.vel_state,  h_grid)
            vl  = _l2norm(s.vel_lambda, h_grid)
            vd  = _l2norm(s.vel_design, h_grid)
            u_norm = _l2norm(s.state, h_grid)
            lsr = u_norm > 0 ? _l2norm(s.lambda .+ s.state, h_grid) / u_norm : 0.0

            _record_lambda!(history, t, J, gs, gl, gd, vs, vl, vd, lsr,
                            s.state, s.lambda, s.design, s.save_fields)

            if s.verbose
                @printf("  step %6d  t = %.4f  J = %.6f  |F_u| = %.3e  |F_λ| = %.3e  |F_θ| = %.3e  ‖λ+u‖/‖u‖ = %.3e\n",
                        step, t, J, gs, gl, gd, lsr)
            end
        end
    end

    return history
end
