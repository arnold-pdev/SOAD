# SymplecticEuler.jl — Semi-implicit (symplectic) Euler integrator
#
# Discretises  m·ẍ + c·ẋ = F  as:
#
#   v_{k+1} = v_k  +  dt · (F(x_k, t_k) − c·v_k) / m     [m > 0]
#   x_{k+1} = x_k  +  dt · v_{k+1}                        (symplectic: updated v)
#
# For m = 0 (first-order / overdamped):
#   x_{k+1} = x_k  +  (dt/c) · F(x_k, t_k)
#   v_{k+1} = 0
#
# FIRE velocity mixing (triggered when sp.restart or when the schedule is a
# FIRESchedule) is applied before the velocity update:
#
#   if sp.restart: v ← 0
#   v ← (1 − α)·v  +  α · ‖v‖ · F̂      [FIRE mixing]
#   then proceed with standard inertial update
#
# This is extracted verbatim from the logic previously in SecondOrderSolver.jl
# and FirstOrderSolver.jl, unified here.

import LinearAlgebra: norm

"""
    SymplecticEuler

Semi-implicit Euler integrator.  Supports m > 0 (inertial), m = 0
(first-order gradient flow), and FIRE velocity mixing.

Compatible with all schedule types including FIRESchedule.
"""
struct SymplecticEuler <: AbstractIntegrator end

function integrate!(::SymplecticEuler, x, v, F_fn, sp, dt, t; is_fire::Bool=false)
    m = sp.mass
    c = sp.damping

    # Evaluate force at current position and time
    F = F_fn(x, t)

    # FIRE restart: zero velocity before mixing
    if sp.restart
        fill!(v, 0.0)
    end

    if is_fire
        # FIRE velocity mixing: v ← (1−c)·v + c·‖v‖·F̂
        # Here sp.damping is the FIRE mixing coefficient α_fire.
        v_norm = norm(v)
        F_norm = norm(F)
        if v_norm > 0 && F_norm > 0
            @. v = (1 - c) * v + c * (v_norm / F_norm) * F
        end
        # FIRE physical update: pure acceleration (no linear damping term)
        @. v += dt * F
        @. x += dt * v

    elseif m > 0
        # Standard inertial (heavy-ball / Nesterov / constant mass)
        # v_{k+1} = v_k + dt·(F − c·v_k)/m
        @. v += dt * (F - c * v) / m
        @. x += dt * v

    else
        # First-order overdamped: x_{k+1} = x_k + (dt/c)·F
        @. x += (dt / c) * F
        fill!(v, 0.0)
    end

    return nothing
end
