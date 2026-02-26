# RK4.jl — Classical fourth-order Runge-Kutta integrator
#
# Integrates  m·ẍ + c(t)·ẋ = F(x, t)  by rewriting as a first-order system
#
#   ẋ = v
#   v̇ = (F(x, t) − c(t)·v) / m
#
# and applying the classical RK4 Butcher tableau (c = [0, 1/2, 1/2, 1]):
#
#   k1 = f(x_n,        v_n,        t_n)
#   k2 = f(x_n+dt/2·k1_x, v_n+dt/2·k1_v, t_n+dt/2)
#   k3 = f(x_n+dt/2·k2_x, v_n+dt/2·k2_v, t_n+dt/2)
#   k4 = f(x_n+dt·k3_x,   v_n+dt·k3_v,   t_n+dt)
#
#   x_{n+1} = x_n + (dt/6)·(k1_x + 2k2_x + 2k3_x + k4_x)
#   v_{n+1} = v_n + (dt/6)·(k1_v + 2k2_v + 2k3_v + k4_v)
#
# where f(x, v, t) = (v,  (F(x,t) − c(t)·v) / m).
#
# Non-autonomous damping
# ----------------------
# c(t) is evaluated at each substage time.  The schedule is queried via
# sp.damping for the current step; for non-autonomous schedules the caller
# should pass the substage damping directly.  In practice, since the schedule
# is only queried once per full step (before calling integrate!), c is held
# constant across the four substages.  This is consistent with the schedule
# being "piecewise constant per step" — acceptable when dt is small relative
# to the timescale on which c varies.
#
# FIRE incompatibility
# --------------------
# FIRE's velocity restart and mixing are inherently discrete (depend on sign
# of F·v at step start).  These do not generalize cleanly to multi-stage
# integration.  Passing is_fire=true raises an error.
#
# First-order (m=0) case
# ----------------------
# RK4 applied to ẋ = (1/c)·F(x,t) — standard RK4 on the first-order ODE.
# Velocity v is unused and zeroed after the step.

"""
    RK4Integrator

Classical fourth-order Runge-Kutta integrator.

Handles both m > 0 (second-order inertial) and m = 0 (first-order).
Not compatible with FIRESchedule (is_fire=true raises an error).
"""
struct RK4Integrator <: AbstractIntegrator end

function integrate!(::RK4Integrator, x, v, F_fn, sp, dt, t; is_fire::Bool=false)
    if is_fire
        error("RK4Integrator is not compatible with FIRESchedule. " *
              "Use SymplecticEuler with FIRESchedule.")
    end

    m = sp.mass
    c = sp.damping

    if m > 0
        _rk4_second_order!(x, v, F_fn, m, c, dt, t)
    else
        _rk4_first_order!(x, v, F_fn, c, dt, t)
    end

    return nothing
end

# Second-order RK4: state = (x, v), derivative = (v, (F - c·v)/m)
function _rk4_second_order!(x, v, F_fn, m, c, dt, t)
    # Allocate stage increments
    k1x = similar(x); k1v = similar(v)
    k2x = similar(x); k2v = similar(v)
    k3x = similar(x); k3v = similar(v)
    k4x = similar(x); k4v = similar(v)

    # Stage 1: (x_n, v_n, t_n)
    F1 = F_fn(x, t)
    @. k1x = v
    @. k1v = (F1 - c * v) / m

    # Stage 2: (x_n + dt/2·k1x, v_n + dt/2·k1v, t_n + dt/2)
    x2 = x .+ (dt/2) .* k1x
    v2 = v .+ (dt/2) .* k1v
    F2 = F_fn(x2, t + dt/2)
    @. k2x = v2
    @. k2v = (F2 - c * v2) / m

    # Stage 3: (x_n + dt/2·k2x, v_n + dt/2·k2v, t_n + dt/2)
    x3 = x .+ (dt/2) .* k2x
    v3 = v .+ (dt/2) .* k2v
    F3 = F_fn(x3, t + dt/2)
    @. k3x = v3
    @. k3v = (F3 - c * v3) / m

    # Stage 4: (x_n + dt·k3x, v_n + dt·k3v, t_n + dt)
    x4 = x .+ dt .* k3x
    v4 = v .+ dt .* k3v
    F4 = F_fn(x4, t + dt)
    @. k4x = v4
    @. k4v = (F4 - c * v4) / m

    # Combine
    @. x += (dt / 6) * (k1x + 2*k2x + 2*k3x + k4x)
    @. v += (dt / 6) * (k1v + 2*k2v + 2*k3v + k4v)
end

# First-order RK4: ẋ = F(x,t)/c
function _rk4_first_order!(x, v, F_fn, c, dt, t)
    k1 = F_fn(x, t) ./ c
    k2 = F_fn(x .+ (dt/2) .* k1, t + dt/2) ./ c
    k3 = F_fn(x .+ (dt/2) .* k2, t + dt/2) ./ c
    k4 = F_fn(x .+ dt .* k3,     t + dt)   ./ c

    @. x += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    fill!(v, 0.0)
end
