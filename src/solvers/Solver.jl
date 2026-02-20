# Solver.jl — Abstract interface for dynamic-relaxation solvers
#
# A solver couples a problem (which provides gradients and BCs) with two
# schedules (one for the state, one for the design) and integrates the
# inertial dynamics forward in time.
#
# The continuous equations being discretised are
#
#   State (fast variable, typically minimised):
#       m_u · ü  +  α_u · u̇  =  −∂U/∂u
#
#   Design (slow variable, here maximised for compliance):
#       m_θ · θ̈  +  α_θ · θ̇  =  +∂J/∂θ
#
# where (m, α) are provided by the respective schedulers.
#
# Special cases recovered from this formulation:
#   m=0, α=1   : L² gradient flow  (first-order)
#   m=1, α=3/t : Su-Boyd-Candès / Nesterov
#   m=1, α=FIRE: FIRE adaptive relaxation
#
# Equilibrium-constrained mode
# ----------------------------
# When `constrained = true`, the state is projected onto the equilibrium
# manifold after every design update:
#
#   θ update: θ ← θ + dt·v_θ
#   State fix: u ← u*(θ)  [exact solve, no velocity on u]
#
# This corresponds to the slow-fast (ε→0) limit of the two-timescale system.
# Numerically it replaces the stiff state ODE with an exact projection, which
# is both more stable and more expensive (per step).
#
# The choice between constrained and joint (unconstrained) dynamics is the
# central experimental axis of this framework.

# ============================================================
#  History (result record)
# ============================================================

"""
    SolverHistory

Records the trajectory of a solve.  Fields are appended every `stride` steps.

Scalar diagnostics are always recorded.  Full field snapshots (state, design,
velocities) are recorded only when `save_fields = true` was passed to the
solver.

Fields
------
Scalar time series (always recorded):
- `t`            : Vector{Float64} — simulation time at each record point
- `objective`    : Vector{Float64} — J(u, θ)
- `grad_state_norm` : Vector{Float64} — ‖∂U/∂u‖_L²
- `grad_design_norm` : Vector{Float64} — ‖∂J/∂θ‖_L²
- `vel_state_norm`  : Vector{Float64} — ‖u̇‖_L²
- `vel_design_norm` : Vector{Float64} — ‖θ̇‖_L²

Field snapshots (recorded when save_fields = true):
- `state_snapshots`  : Vector{Vector{Float64}}  — list of state vectors
- `design_snapshots` : Vector{Vector{Float64}}  — list of design vectors
"""
mutable struct SolverHistory
    # Time stamps
    t                ::Vector{Float64}

    # Scalar diagnostics
    objective        ::Vector{Float64}
    grad_state_norm  ::Vector{Float64}
    grad_design_norm ::Vector{Float64}
    vel_state_norm   ::Vector{Float64}
    vel_design_norm  ::Vector{Float64}

    # Optional field snapshots
    state_snapshots  ::Vector{Vector{Float64}}
    design_snapshots ::Vector{Vector{Float64}}
end

SolverHistory() = SolverHistory(
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
    Vector{Float64}[], Vector{Float64}[]
)

function _record!(h::SolverHistory, t, J, gs, gd, vs, vd, state, design, save_fields)
    push!(h.t,                t)
    push!(h.objective,        J)
    push!(h.grad_state_norm,  gs)
    push!(h.grad_design_norm, gd)
    push!(h.vel_state_norm,   vs)
    push!(h.vel_design_norm,  vd)
    if save_fields
        push!(h.state_snapshots,  copy(state))
        push!(h.design_snapshots, copy(design))
    end
end

# L² norm on a staggered grid with cell spacing h
_l2norm(v::AbstractVector, h::Float64) = sqrt(h * sum(abs2, v))

# ============================================================
#  Abstract solver
# ============================================================

"""
    AbstractSolver

Supertype for all dynamic-relaxation solvers.

Concrete subtypes combine a problem with state and design schedules and
implement the core time-integration loop.

## Required interface

    solve!(solver) -> SolverHistory

Run the solver to completion and return the recorded history.

## Constructor convention

Concrete solvers should accept keyword arguments:

    XxxSolver(
        problem,
        state_schedule,
        design_schedule;
        dt          = nothing,   # if nothing, use stable_dt(problem, ...)
        T           = 10.0,      # total simulation time
        stride      = 10,        # record every stride steps
        save_fields = false,     # whether to store full field snapshots
        constrained = false,     # equilibrium-constrained design-only mode
        verbose     = false
    )

See `GradientFlowSolver` and `InertialSolver` for concrete implementations.
"""
abstract type AbstractSolver end

"""
    solve!(solver::AbstractSolver) -> SolverHistory

Run the dynamic relaxation solver and return a `SolverHistory` with all
recorded diagnostics.  Mutates the solver's internal state/design fields.
"""
function solve!(solver::AbstractSolver)
    error("solve! not implemented for $(typeof(solver))")
end

"""
    current_state(solver::AbstractSolver) -> Vector{Float64}

Return the solver's current state vector.
"""
function current_state(solver::AbstractSolver)
    error("current_state not implemented for $(typeof(solver))")
end

"""
    current_design(solver::AbstractSolver) -> Vector{Float64}

Return the solver's current design vector.
"""
function current_design(solver::AbstractSolver)
    error("current_design not implemented for $(typeof(solver))")
end
