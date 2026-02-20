# FIRESchedule.jl — Fast Inertial Relaxation Engine schedule
#
# FIRE (Bitzek et al., PRL 2006) is an adaptive damping scheme developed for
# molecular dynamics but widely applicable to structural optimisation.  The
# core idea is:
#
#   1. If the current "power"  P = −∇f · v > 0  (gradient and velocity agree),
#      reduce damping (let momentum accumulate) and nudge velocity toward the
#      gradient direction.
#   2. If  P ≤ 0  (gradient and velocity disagree), restart: zero the velocity,
#      reset to maximum damping, and start accumulating momentum again from
#      scratch.
#
# The parameter update rules at each step:
#   - Always:  v ← (1 - α)·v - α·|v|·∇̂f   (mix velocity toward −∇f)
#   - If P > 0 and we've waited N_min steps since last restart:
#       dt   ← min(dt · f_inc, dt_max)
#       α    ← α · f_α               (reduce mixing coefficient)
#       mass ← mass · f_inc          (not standard; omitted here)
#   - If P ≤ 0:
#       dt   ← dt · f_dec
#       α    ← α_start               (reset mixing)
#       emit restart signal
#
# In this framework we *do not* update dt inside the scheduler (dt is the
# solver's concern).  Instead we encode the adaptation as a time-varying
# (mass=1, damping=α(t)) with a restart flag, which is the natural
# translation into our (m, α, restart) language.
#
# Reference: Bitzek, Koskinen, Gähler, Moseler, Gumbsch.
# "Structural Relaxation Made Simple." PRL 97, 170201 (2006).

"""
    FIRESchedule(; α_start=0.1, f_α=0.99, N_min=5)

Adaptive FIRE-style scheduler.

Parameters
----------
- `α_start` : initial mixing coefficient (∈ (0, 1))
- `f_α`     : factor by which α decreases on positive-power steps (< 1)
- `N_min`   : minimum steps between restarts (guards against rapid oscillation)

The scheduler maintains internal state:
- `α_current` : current mixing coefficient
- `steps_since_restart` : steps since last velocity reset

On each `step!` call:
- Computes power  P = dot(grad, vel)  (note: sign depends on ascent/descent
  convention; the solver should pass `grad` with the sign appropriate to the
  *descent* direction, i.e. always pass −∇f for a minimiser or +∇f for a
  maximiser so that P > 0 means "moving in the right direction").
- Returns `ScheduleParams(mass=1, damping=α, restart)`.

The mass is always 1; all adaptation appears through damping and restart.

Examples
--------
```julia
s = FIRESchedule()
s = FIRESchedule(α_start=0.2, f_α=0.995, N_min=10)
```
"""
mutable struct FIRESchedule <: AbstractSchedule
    # Hyperparameters (fixed after construction)
    α_start ::Float64
    f_α     ::Float64
    N_min   ::Int

    # Mutable internal state
    α_current          ::Float64
    steps_since_restart ::Int
end

function FIRESchedule(; α_start=0.1, f_α=0.99, N_min=5)
    0 < α_start < 1 || throw(ArgumentError("α_start must be in (0,1)"))
    0 < f_α     < 1 || throw(ArgumentError("f_α must be in (0,1)"))
    N_min ≥ 1       || throw(ArgumentError("N_min must be ≥ 1"))
    FIRESchedule(α_start, f_α, N_min, α_start, 0)
end

function reset!(s::FIRESchedule)
    s.α_current           = s.α_start
    s.steps_since_restart = 0
    return s
end

"""
    step!(s::FIRESchedule, k, t, grad, vel) -> ScheduleParams

Advance the FIRE scheduler.

The `grad` argument should be the *negative* of the force direction:
- For a state variable being minimised, pass +∂U/∂u (so P = −∇f·v > 0 when
  the velocity is going downhill).
- For a design variable being maximised, pass −∂J/∂θ (same convention).

In practice, the solvers in this framework pass the raw gradient with the
solver's sign convention, so each problem's solver wrapper is responsible for
negating if needed.  The convention here is:

    P = dot(grad, vel)   with  grad = force acting on the variable

i.e. P > 0 means the velocity is aligned with the force.
"""
function step!(s::FIRESchedule, k, t, grad, vel)
    s.steps_since_restart += 1

    # Power: P > 0 ⟺ velocity aligned with gradient force
    P = dot(grad, vel)

    if P > 0 && s.steps_since_restart ≥ s.N_min
        # Good direction: reduce mixing coefficient (less correction toward ∇f)
        s.α_current = s.α_current * s.f_α
        return ScheduleParams(1.0, s.α_current, false)
    elseif P ≤ 0
        # Bad direction: restart
        s.α_current           = s.α_start
        s.steps_since_restart = 0
        return ScheduleParams(1.0, s.α_current, true)
    else
        # P > 0 but haven't waited N_min steps — hold current α
        return ScheduleParams(1.0, s.α_current, false)
    end
end

