# Integrator.jl — Abstract interface for time-integration kernels
#
# An integrator advances a single variable (x, v) by one timestep given a
# force function F(x, t) and schedule parameters (mass, damping).  It is
# deliberately problem-agnostic: it knows nothing about beams or objectives,
# only about the ODE structure
#
#   m · ẍ  +  c(t) · ẋ  =  F(x, t)
#
# Solvers compose an integrator with schedules and a problem to produce a
# complete algorithm.
#
# Interface
# ---------
# Every concrete integrator must implement:
#
#   integrate!(integ, x, v, F_fn, sp, dt, t)
#
# where
#   x     : current position (mutated in-place)
#   v     : current velocity (mutated in-place)
#   F_fn  : callable F_fn(x, t) -> force vector (same shape as x)
#   sp    : ScheduleParams — (mass, damping, restart) at the current step
#   dt    : timestep
#   t     : current time (start of step; needed for non-autonomous F_fn, c(t))
#
# Notes on non-autonomy
# ---------------------
# Both F and c(t) can vary with time.  SymplecticEuler evaluates F once at
# (x_k, t_k).  RK4 evaluates F at the four Butcher substage times; the
# schedule's damping at each substage is obtained by calling the schedule's
# damping_at(t) method if available, otherwise the value in sp is used
# throughout the step.
#
# FIRE compatibility
# ------------------
# FIRE velocity mixing is a discrete operation that depends on the sign of
# F·v at the start of the step.  It is implemented inside SymplecticEuler
# via the standard restart/mix logic.  RK4Integrator will throw an error if
# paired with FIRESchedule, since the restart semantics do not compose
# cleanly with multi-stage integration.

"""
    AbstractIntegrator

Supertype for all time-integration kernels.

Concrete subtypes implement `integrate!(integ, x, v, F_fn, sp, dt, t)`.
"""
abstract type AbstractIntegrator end

"""
    integrate!(integ, x, v, F_fn, sp, dt, t)

Advance position `x` and velocity `v` by one step of size `dt`, starting at
time `t`, using force function `F_fn(x, t) -> Vector` and schedule parameters
`sp::ScheduleParams`.

Mutates `x` and `v` in place.  Returns nothing.
"""
function integrate!(integ::AbstractIntegrator, x, v, F_fn, sp, dt, t)
    error("integrate! not implemented for $(typeof(integ))")
end
