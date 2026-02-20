# NesterovSchedule.jl — Continuous-time Nesterov / Su-Boyd-Candès schedule
#
# The Su-Boyd-Candès (SBC, 2016) ODE that achieves O(1/k²) convergence is
#
#   ẍ + (r/t) ẋ + ∇f(x) = 0
#
# with r = 3 recovering the Nesterov optimal rate for convex objectives.
# In our (m, α) language this corresponds to
#
#   m = 1,   α(t) = r / max(t, ε)
#
# where ε > 0 is a small floor that avoids the singularity at t = 0.
#
# Reference: Su, Boyd, Candès. "A Differential Equation for Modeling
# Nesterov's Accelerated Gradient Method." JMLR 2016.

"""
    NesterovSchedule(; r=3.0, t_floor=1e-6)

Continuous-time Nesterov schedule with damping coefficient α(t) = r/t.

Parameters
----------
- `r`       : damping exponent (r = 3 achieves O(1/k²) rate for r ≥ 3)
- `t_floor` : small positive value used as a lower bound on t to avoid the
              singularity at t = 0 (default: 1e-6)

The mass is fixed at 1.  Damping decreases monotonically as 1/t, so early
iterates are heavily damped and later iterates retain more momentum.

No restart signals are emitted (the SBC ODE is globally well-posed for t > 0).

Examples
--------
```julia
s = NesterovSchedule()          # standard r=3 schedule
s = NesterovSchedule(r=2.0)     # softer momentum (convergence O(1/k^{4/3}))
```
"""
struct NesterovSchedule <: AbstractSchedule
    r       ::Float64
    t_floor ::Float64
end

NesterovSchedule(; r=3.0, t_floor=1e-6) = NesterovSchedule(r, t_floor)

function step!(s::NesterovSchedule, k, t, grad, vel)
    α = s.r / max(t, s.t_floor)
    return ScheduleParams(1.0, α, false)
end
