# ConstantSchedule.jl — Fixed mass and damping (no adaptation)
#
# The simplest possible scheduler: returns the same (mass, damping) at every
# step.  This covers two important special cases:
#
#   mass = 0               : pure L² gradient flow (first-order)
#   mass > 0, damping > 0  : heavy-ball / constant-momentum inertial flow
#
# There is never a restart signal.

"""
    ConstantSchedule(; mass=0.0, damping=1.0)

Stateless scheduler returning fixed (mass, damping) at every step.

- `mass = 0`  → first-order gradient flow  (default)
- `mass > 0`  → constant-mass inertial flow (heavy-ball)

No restart signals are ever emitted.

Examples
--------
```julia
# Pure gradient descent
s = ConstantSchedule()

# Heavy-ball with mass 0.1 and damping 0.5
s = ConstantSchedule(mass=0.1, damping=0.5)
```
"""
struct ConstantSchedule <: AbstractSchedule
    params ::ScheduleParams
end

function ConstantSchedule(; mass=0.0, damping=1.0)
    mass   ≥ 0  || throw(ArgumentError("mass must be ≥ 0"))
    damping > 0 || throw(ArgumentError("damping must be > 0"))
    ConstantSchedule(ScheduleParams(mass, damping, false))
end

step!(s::ConstantSchedule, k, t, grad, vel) = s.params
