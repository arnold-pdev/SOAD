# Schedule.jl — Abstract interface for parameter schedules
#
# A schedule controls the damping and mass (inertia) coefficients that govern
# the dynamics of state and design variables over the course of a run.  It is
# intentionally problem-agnostic: it knows nothing about beams or elasticity,
# only about the abstract structure of the integrator.
#
# The central design decision (chosen by the user) is:
#   - State and design have *independent* schedule instances.
#   - Each scheduler step returns a `ScheduleParams` struct plus an optional
#     restart signal.  The solver acts on the restart signal by zeroing the
#     velocity of the corresponding variable.
#
# Scheduler "language"
# --------------------
# The second-order inertial ODE for a variable x is written as
#
#   m · ẍ  +  α · ẋ  =  ±∇f(x)
#
# where
#   m  (mass)     ≥ 0  — inertia; m=0 collapses to first-order gradient flow
#   α  (damping)  > 0  — dissipation rate
#
# In explicit time-stepping this becomes
#
#   vₖ₊₁ = (1 - α·dt/m)·vₖ  -  (dt/m)·∇f(xₖ)     [m > 0]
#   xₖ₊₁ = xₖ + dt·vₖ₊₁
#
# or, for m=0 (pure gradient flow):
#
#   xₖ₊₁ = xₖ  -  (dt/α)·∇f(xₖ)
#
# The scheduler provides (m, α) at each step, and both can vary over time.
# Special cases:
#   Gradient descent         : m = 0,   α = 1
#   Heavy-ball               : m > 0,   α = const > 0
#   Nesterov (SBC continuous): m = 1,   α = 3/t  (t = current time)
#   FIRE                     : m, α updated adaptively; restart when P = F·v < 0
#
# This design lets you implement any of these as a concrete scheduler.

# ============================================================
#  Return type
# ============================================================

"""
    ScheduleParams

Parameters returned by a scheduler at each step.

Fields
------
- `mass`    : Float64  — inertia coefficient m ≥ 0.  Set to 0 for first-order.
- `damping` : Float64  — dissipation coefficient α > 0.
- `restart` : Bool     — if true, the solver should zero this variable's velocity
                         before applying the update (FIRE-style restart).
"""
struct ScheduleParams
    mass    ::Float64
    damping ::Float64
    restart ::Bool
end

ScheduleParams(mass, damping) = ScheduleParams(mass, damping, false)

# ============================================================
#  Abstract scheduler
# ============================================================

"""
    AbstractSchedule

Supertype for all parameter schedulers.

A scheduler is a stateful object that, given the current iteration index `k`,
elapsed time `t`, and gradient/velocity information, returns a `ScheduleParams`
struct controlling the next integrator step.

## Required interface

Every concrete subtype must implement:

    step!(sched, k, t, grad, vel) -> ScheduleParams

where
- `k`    : Int     — iteration counter (1-indexed)
- `t`    : Float64 — elapsed (simulation) time
- `grad` : AbstractVector — current gradient ∇f (or functional derivative)
- `vel`  : AbstractVector — current velocity (zeros for first-order methods)

The function may mutate the scheduler's internal state (e.g. FIRE bookkeeping).

## Optional interface

    reset!(sched)   — reset internal state to initial values

See `ConstantSchedule`, `NesterovSchedule`, and `FIRESchedule` for examples.
"""
abstract type AbstractSchedule end

"""
    step!(sched, k, t, grad, vel) -> ScheduleParams

Advance the scheduler by one step and return the parameters for the current
integrator update.  May mutate `sched`'s internal state.
"""
function step!(sched::AbstractSchedule, k, t, grad, vel)
    error("step! not implemented for $(typeof(sched))")
end

"""
    reset!(sched::AbstractSchedule)

Reset the scheduler to its initial state.  Default is a no-op for stateless
schedules.
"""
reset!(sched::AbstractSchedule) = nothing
