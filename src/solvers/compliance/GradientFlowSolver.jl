# FirstOrderSolver.jl — Pure gradient flow (first-order dynamics)
#
# This solver integrates the L² gradient flow
#
#   u̇  = −(1/α_u) · ∂U/∂u
#   θ̇  = +(1/α_θ) · ∂J/∂θ
#
# using explicit Euler.  The (1/α) factor absorbs the step-size scaling;
# the `ConstantSchedule(mass=0, damping=α)` convention means the effective
# step is dt/α per unit time.
#
# Equilibrium-constrained mode
# ----------------------------
# When `constrained = true` the state equation is replaced by an exact solve:
# after each design update, u is set to u*(θ) = equilibrium_state(problem, θ).
# The state velocity is unused (but still returned as zero in history).
# This is the slow-fast limit: state timescale → 0.
#
# The constrained and unconstrained modes are both first-class; they can be
# run with the same scheduler objects and the same output structure, making
# them straightforward to compare.

import LinearAlgebra: norm

"""
    GradientFlowSolver

First-order (no inertia) dynamic-relaxation solver.

Both state and design use a `ConstantSchedule(mass=0)` by default, but any
`AbstractSchedule` can be passed — the mass field is simply ignored.

Constructor
-----------
```julia
s = GradientFlowSolver(
    problem,
    state_schedule,
    design_schedule;
    dt          = nothing,   # auto from stable_dt if nothing
    T           = 10.0,
    stride      = 10,
    save_fields = false,
    constrained = false,     # if true: state always at equilibrium u*(θ)
    verbose     = false
)
```

Dynamics
--------
Unconstrained (joint):

    uₖ₊₁ = uₖ − (dt/αᵤ) · ∂U/∂u
    θₖ₊₁ = θₖ + (dt/αθ) · ∂J/∂θ

Constrained (equilibrium-projected, slow-fast limit):

    θₖ₊₁ = θₖ + (dt/αθ) · ∂J/∂θ
    uₖ₊₁ = u*(θₖ₊₁)              [exact equilibrium solve]

The gradient ∂J/∂θ is evaluated at the current (u, θ) in both modes.
"""
mutable struct GradientFlowSolver{P, SS, DS} <: AbstractSolver
    problem        ::P
    state_sched    ::SS
    design_sched   ::DS

    # Current iterate
    state          ::Vector{Float64}
    design         ::Vector{Float64}

    # Solver settings
    dt             ::Float64
    T              ::Float64
    stride         ::Int
    save_fields    ::Bool
    constrained    ::Bool
    verbose        ::Bool
end

function GradientFlowSolver(
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
        # Initialize state at equilibrium for the initial design
        state0 = equilibrium_state(problem, design0)
    end

    enforce_state_bcs!(problem, state0)
    enforce_design_bcs!(problem, design0)

    dt_actual = (dt === nothing) ? stable_dt(problem, state0, design0) : Float64(dt)

    GradientFlowSolver(
        problem, state_sched, design_sched,
        state0, design0,
        dt_actual, Float64(T), stride, save_fields, constrained, verbose
    )
end

current_state(s::GradientFlowSolver)  = s.state
current_design(s::GradientFlowSolver) = s.design

"""
    solve!(s::GradientFlowSolver) -> SolverHistory

Run the gradient flow to time T with timestep dt.
Records diagnostics every `stride` steps.
"""
function solve!(s::GradientFlowSolver)
    h       = s.problem.grid.h   # spatial grid spacing for L² norms
    history = SolverHistory()
    n_steps = round(Int, s.T / s.dt)
    k       = 0
    t       = 0.0

    # Dummy velocities (zero, unused in first-order mode — but needed for
    # scheduler interface)
    vel_state  = zeros(state_dim(s.problem))
    vel_design = zeros(design_dim(s.problem))

    for step in 1:n_steps
        k += 1
        t  = step * s.dt

        # 1. Compute gradients at current (state, design)
        grad_state, grad_design = energy_gradients(s.problem, s.state, s.design)

        # 2. Query schedulers (mass is ignored in first-order mode)
        sp_s = step!(s.state_sched,  k, t, grad_state,  vel_state)
        sp_d = step!(s.design_sched, k, t, grad_design, vel_design)

        # 3. Update design (ascent on J)
        s.design .+= (s.dt / sp_d.damping) .* grad_design
        enforce_design_bcs!(s.problem, s.design)

        # 4. Update state
        if s.constrained
            # Slow-fast limit: project onto equilibrium manifold
            s.state = equilibrium_state(s.problem, s.design)
        else
            # Joint dynamics: gradient descent on U
            s.state .-= (s.dt / sp_s.damping) .* grad_state
        end
        enforce_state_bcs!(s.problem, s.state)

        # 5. Record diagnostics every `stride` steps
        if step % s.stride == 0
            J  = objective(s.problem, s.state, s.design)
            gs = _l2norm(grad_state,  h)
            gd = _l2norm(grad_design, h)
            vs = 0.0   # first-order has no velocity
            vd = 0.0
            _record!(history, t, J, gs, gd, vs, vd,
                     s.state, s.design, s.save_fields)

            if s.verbose
                @printf("  step %6d  t = %.4f  J = %.6f  |∇u| = %.3e  |∇θ| = %.3e\n",
                        step, t, J, gs, gd)
            end
        end
    end

    return history
end
