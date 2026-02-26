# SecondOrderSolver.jl — Inertial (second-order) dynamic relaxation
#
# Integrates the inertial equations
#
#   m_u · ü  +  α_u · u̇  =  F_u  := −∂U/∂u
#   m_θ · θ̈  +  α_θ · θ̇  =  F_θ  := +∂J/∂θ
#
# using a symplectic (velocity-Verlet-like) explicit scheme for m > 0, and
# falling back to the first-order update when m = 0.
#
# Discretization of the inertial ODE
# ------------------------------------
# Re-write as a first-order system by introducing velocities v = ẋ:
#
#   vₖ₊₁ = vₖ  +  dt · (F(xₖ) − α·vₖ) / m       [m > 0]
#   xₖ₊₁ = xₖ  +  dt · vₖ₊₁                     (symplectic: use updated v)
#
# This is the semi-implicit Euler (symplectic Euler) method.  It conserves
# a modified energy in the undamped case (α=0), which is important for FIRE
# where α is intermittently small.
#
# When m = 0 (returned by a first-order scheduler):
#   xₖ₊₁ = xₖ  +  (dt/α) · F(xₖ)               (pure gradient step)
#   vₖ₊₁ = 0                                    (no velocity to carry)
#
# FIRE velocity mixing
# --------------------
# FIRE modifies the velocity at each step by mixing it toward the gradient:
#
#   v ← (1 − α_fire)·v  +  α_fire · |v| · F̂
#
# where F̂ = F/|F| is the unit gradient force and α_fire is the FIRE mixing
# coefficient (returned as `damping` by FIRESchedule).  When FIRESchedule
# emits `restart = true`, the velocity is zeroed before the mixing step.
#
# This mixing is *in addition to* the standard velocity update above.  The
# FIRESchedule is designed to work with mass=1 (it always returns mass=1),
# so the combined update is:
#
#   if restart: v ← 0
#   v ← (1 − α)·v  +  α · |v| · F̂   [FIRE mixing]
#   v ← v + dt · F                     [gradient acceleration, α term absorbed]
#   x ← x + dt · v
#
# NOTE: FIRE normally also adapts the timestep dt internally.  Here we keep dt
# fixed (it's the solver's concern) and let the schedule control only (α, restart).
#
# Equilibrium-constrained mode
# ----------------------------
# When `constrained = true`:
#   - The state is always projected to u*(θ) after the design update.
#   - The state inertia (vel_state) is unused and zeroed.
#   - Only the design has inertia.
# This is the slow-fast limit: state "mass" effectively → 0 via exact projection.

import LinearAlgebra: norm, dot

"""
    InertialSolver

Second-order (inertial) dynamic-relaxation solver.

Integrates the coupled inertial equations for state and design using the
symplectic Euler method.  Mass and damping are provided step-by-step by
independent schedulers for state and design.

Supports FIRE-style velocity mixing automatically when the scheduler returns
`restart = true`.

Constructor
-----------
```julia
s = InertialSolver(
    problem,
    state_schedule,
    design_schedule;
    dt          = nothing,
    T           = 10.0,
    stride      = 10,
    save_fields = false,
    constrained = false,
    verbose     = false
)
```

Dynamics (per step, for variable x with velocity v, force F, params (m,α)):
- If m > 0:
    1. (FIRE restart) if restart: v ← 0
    2. (FIRE mix)    v ← (1−α)·v + α·‖v‖·F̂   [only if using FIRESchedule]
    3. v ← v + dt·F/m  − dt·(α/m)·v           [inertial update]
    4. x ← x + dt·v
- If m = 0:
    x ← x + (dt/α)·F                           [first-order step]
    v ← 0

The FIRE mixing step (2) is triggered only when the scheduler is a `FIRESchedule`
(detected via duck-typing: `hasproperty(sched, :α_current)`).
"""
mutable struct InertialSolver{P, SS, DS} <: AbstractSolver
    problem        ::P
    state_sched    ::SS
    design_sched   ::DS

    # Current iterate
    state          ::Vector{Float64}
    design         ::Vector{Float64}
    vel_state      ::Vector{Float64}
    vel_design     ::Vector{Float64}

    # Settings
    dt             ::Float64
    T              ::Float64
    stride         ::Int
    save_fields    ::Bool
    constrained    ::Bool
    verbose        ::Bool
end

function InertialSolver(
    problem,
    state_sched,
    design_sched;
    dt          = nothing,
    T           = 10.0,
    stride      = 10,
    save_fields = false,
    constrained = false,
    verbose     = false
)
    state0  = initial_state(problem)
    design0 = initial_design(problem)

    if constrained
        state0 = equilibrium_state(problem, design0)
    end

    enforce_state_bcs!(problem, state0)
    enforce_design_bcs!(problem, design0)

    dt_actual = (dt === nothing) ? stable_dt(problem, state0, design0) : Float64(dt)

    InertialSolver(
        problem, state_sched, design_sched,
        state0, design0,
        zeros(length(state0)), zeros(length(design0)),
        dt_actual, Float64(T), stride, save_fields, constrained, verbose
    )
end

current_state(s::InertialSolver)  = s.state
current_design(s::InertialSolver) = s.design

# ---- core update for one variable ------------------------------------------

# Apply one inertial step to variable x with velocity v, force F, params sp.
# Mutates x and v in place.
# `is_fire` : whether to apply FIRE velocity mixing (based on schedule type)
function _inertial_update!(x, v, F, sp::ScheduleParams, dt, is_fire::Bool)
    m = sp.mass
    α = sp.damping

    if sp.restart
        fill!(v, 0.0)
    end

    if is_fire
        # FIRE velocity mixing: v ← (1−α)·v + α·‖v‖·F̂
        v_norm = norm(v)
        F_norm = norm(F)
        if v_norm > 0 && F_norm > 0
            @. v = (1 - α) * v + α * (v_norm / F_norm) * F
        end
        # In FIRE the "damping" parameter is the mixing coefficient α_fire, not
        # a physical damping.  The physical update is just acceleration: v += F·dt.
        @. v += dt * F
        @. x += dt * v
    elseif m > 0
        # Standard inertial update (semi-implicit Euler)
        # v_{k+1} = v_k + dt·(F − α·v_k) / m
        @. v += dt * (F - α * v) / m
        @. x += dt * v
    else
        # First-order: x_{k+1} = x_k + (dt/α)·F, v = 0
        @. x += (dt / α) * F
        fill!(v, 0.0)
    end
end

# Detect whether a schedule is a FIRESchedule (duck-typed)
_is_fire(sched) = hasproperty(sched, :α_current)

"""
    solve!(s::InertialSolver) -> SolverHistory
"""
function solve!(s::InertialSolver)
    h       = s.problem.grid.h
    history = SolverHistory()
    n_steps = round(Int, s.T / s.dt)
    k       = 0
    t       = 0.0

    is_fire_state  = _is_fire(s.state_sched)
    is_fire_design = _is_fire(s.design_sched)

    for step in 1:n_steps
        k += 1
        t  = step * s.dt

        # 1. Gradients
        grad_state, grad_design = energy_gradients(s.problem, s.state, s.design)

        # Forces: state is minimized, design is maximized
        F_state  = -grad_state    # force on state  = −∇U
        F_design =  grad_design   # force on design = +∇J

        # 2. Query schedulers (pass force vectors for FIRE power computation)
        sp_s = step!(s.state_sched,  k, t, F_state,  s.vel_state)
        sp_d = step!(s.design_sched, k, t, F_design, s.vel_design)

        # 3. Update design
        _inertial_update!(s.design, s.vel_design, F_design, sp_d, s.dt, is_fire_design)
        enforce_design_bcs!(s.problem, s.design)

        # 4. Update state
        if s.constrained
            # Slow-fast limit: discard state velocity, project onto u*(θ)
            fill!(s.vel_state, 0.0)
            s.state = equilibrium_state(s.problem, s.design)
        else
            _inertial_update!(s.state, s.vel_state, F_state, sp_s, s.dt, is_fire_state)
        end
        enforce_state_bcs!(s.problem, s.state)

        # 5. Diagnostics
        if step % s.stride == 0
            J  = objective(s.problem, s.state, s.design)
            gs = _l2norm(grad_state,    h)
            gd = _l2norm(grad_design,   h)
            vs = _l2norm(s.vel_state,   h)
            vd = _l2norm(s.vel_design,  h)
            _record!(history, t, J, gs, gd, vs, vd,
                     s.state, s.design, s.save_fields)

            if s.verbose
                @printf("  step %6d  t = %.4f  J = %.6f  |∇u| = %.3e  |∇θ| = %.3e  |vθ| = %.3e\n",
                        step, t, J, gs, gd, vd)
            end
        end
    end

    return history
end
