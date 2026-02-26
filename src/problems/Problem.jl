# Problem.jl — Abstract interface for optimization problems
#
# Every concrete problem (Timoshenko beam, 2D elasticity, …) must implement
# this interface. The solvers in this framework operate exclusively through
# these methods, so swapping in a new problem requires no changes to solver
# code.
#
# Terminology used throughout the framework:
#   state  — the "fast" primal variable, e.g. (w, ψ) for a beam.
#             In the continuous limit this satisfies an equilibrium PDE.
#   design — the "slow" design variable, e.g. material orientation θ(x).
#             This is the quantity we are ultimately optimizing.
#
# The framework supports two dynamics regimes:
#   Full (joint) dynamics : state and design evolve simultaneously via
#       inertial / gradient-flow ODEs. Useful for studying the coupled
#       two-timescale behavior.
#   Equilibrium-constrained : state is always projected onto the equilibrium
#       manifold {F(u,θ)=0} after each design step. This is the slow-fast
#       limit where the state timescale → 0. A different (typically implicit)
#       integrator is often needed here for stability.

"""
    AbstractProblem

Supertype for all optimization problems in the adaptive-dynamics framework.

A concrete subtype must represent a structural optimization problem in which
    J(θ) = U(u*(θ), θ) + R(θ)
is minimized over the design field θ, where u* is the equilibrium state
satisfying the primal equations F(u*, θ) = 0.

## Required interface

Implement the following methods for every concrete `AbstractProblem`:

| Method                                | Returns                          |
|---------------------------------------|----------------------------------|
| `state_dim(p)`                        | Int — length of state vector     |
| `design_dim(p)`                       | Int — length of design vector    |
| `initial_state(p)`                    | Vector — feasible starting point |
| `initial_design(p)`                   | Vector — feasible starting point |
| `energy_gradients(p, state, design)`  | (∂U/∂u, ∂J/∂θ)                  |
| `equilibrium_state(p, design)`        | u*(θ) — exact state solve        |
| `objective(p, state, design)`         | scalar J value                   |
| `enforce_state_bcs!(p, state)`        | mutate state in-place            |
| `enforce_design_bcs!(p, design)`      | mutate design in-place           |
| `stable_dt(p, state, design)`        | CFL-safe timestep estimate       |

## Optional interface

| Method                   | Default behavior                     |
|--------------------------|---------------------------------------|
| `grid_points(p)`         | Returns `nothing` (problem-specific)  |
| `problem_name(p)`        | Returns `"AbstractProblem"`           |

See `TimoshenkoProblem` for a fully worked concrete implementation.
"""
abstract type AbstractProblem end

# ---------------------------------------------------------------------------
# Mandatory interface — must be overridden
# ---------------------------------------------------------------------------

"""
    state_dim(p::AbstractProblem) -> Int

Total length of the state vector (all DOFs concatenated into one flat vector).
"""
function state_dim(p::AbstractProblem)
    error("state_dim not implemented for $(typeof(p))")
end

"""
    design_dim(p::AbstractProblem) -> Int

Total length of the design vector.
"""
function design_dim(p::AbstractProblem)
    error("design_dim not implemented for $(typeof(p))")
end

"""
    initial_state(p::AbstractProblem) -> Vector{Float64}

Return a feasible initial state vector satisfying the boundary conditions.
"""
function initial_state(p::AbstractProblem)
    error("initial_state not implemented for $(typeof(p))")
end

"""
    initial_design(p::AbstractProblem) -> Vector{Float64}

Return a feasible initial design vector satisfying the design boundary
conditions (e.g. Neumann BCs on orientation).
"""
function initial_design(p::AbstractProblem)
    error("initial_design not implemented for $(typeof(p))")
end

"""
    energy_gradients(p, state, design) -> (grad_state, grad_design)

Compute the gradients of the regularized objective J with respect to the
state and the design.

    grad_state  = ∂U/∂u   (shape: state_dim(p))
    grad_design = ∂J/∂θ   (shape: design_dim(p))

These are the L²-Riesz representatives of the functional derivatives, i.e.
they are already the vectors that enter the gradient-flow ODEs

    u̇  = -grad_state
    θ̇  = +grad_design   (maximization convention for compliance)

or the corresponding inertial generalisations.

The sign convention here is: the solver **subtracts** grad_state from the
state velocity and **adds** grad_design to the design velocity. Problems
where the signs differ should negate the returned vectors accordingly.
"""
function energy_gradients(p::AbstractProblem, state, design)
    error("energy_gradients not implemented for $(typeof(p))")
end

"""
    equilibrium_state(p, design) -> Vector{Float64}

Compute the exact equilibrium state u*(θ) for a given design θ by solving
the primal PDE F(u*, θ) = 0 analytically or numerically.

Used by the equilibrium-constrained solver to project the state back onto
the equilibrium manifold after each design update.
"""
function equilibrium_state(p::AbstractProblem, design)
    error("equilibrium_state not implemented for $(typeof(p))")
end

"""
    objective(p, state, design) -> Float64

Return the scalar value of the regularized objective J(u, θ).
"""
function objective(p::AbstractProblem, state, design)
    error("objective not implemented for $(typeof(p))")
end

"""
    enforce_state_bcs!(p, state)

Enforce Dirichlet (and any other essential) boundary conditions on the state
vector **in place**. Called after every state update step.
"""
function enforce_state_bcs!(p::AbstractProblem, state)
    error("enforce_state_bcs! not implemented for $(typeof(p))")
end

"""
    enforce_design_bcs!(p, design)

Enforce boundary conditions on the design vector **in place** (e.g. wrap
angles to [0, π), clamp thickness to physical bounds). Called after every
design update step.
"""
function enforce_design_bcs!(p::AbstractProblem, design)
    error("enforce_design_bcs! not implemented for $(typeof(p))")
end

"""
    stable_dt(p, state, design) -> Float64

Return an estimate of the largest timestep that is CFL-stable for an explicit
integrator on this problem. Used by the solver as an upper bound when the
user does not specify `dt` manually.
"""
function stable_dt(p::AbstractProblem, state, design)
    error("stable_dt not implemented for $(typeof(p))")
end

# ---------------------------------------------------------------------------
# General (3-player λ) interface — required for LambdaSolver
# ---------------------------------------------------------------------------
#
# These methods expose the operator-level structure of the PDE that the
# compliance-only solvers never need.  They have no default implementations;
# a problem that does not implement them cannot be used with LambdaSolver.
#
# Mathematical context
# --------------------
# The Lagrangian for the general problem is
#
#   L(u, ρ, λ) = J(u, ρ) + ⟨λ, R(u, ρ)⟩
#
# where R(u, ρ) = A(ρ)u − f is the equilibrium residual.  The three-player
# inertial dynamics are
#
#   m_u · ü  +  c_u(t) · u̇  =  −∂J/∂u − A(ρ)ᵀλ  =: F_u
#   m_λ · λ̈  +  c_λ(t) · λ̇  =  −∂J/∂u − A(ρ)λ   =: F_λ
#   m_ρ · ρ̈  +  c_ρ(t) · ρ̇  =  −∂J/∂ρ − (∂R/∂ρ)ᵀλ =: F_ρ
#
# At the saddle point F_u = F_λ = F_ρ = 0, recovering the adjoint equation
# A(ρ)λ = −∂J/∂u and the design optimality condition.
#
# For self-adjoint A (linear elasticity), A = Aᵀ, so F_u and F_λ share the
# same form; both require apply_stiffness(p, ρ, λ) = A(ρ)λ.
#
# For compliance J = ⟨f, u⟩:  ∂J/∂u = f (the load vector, constant).
# For general J(u, ρ):         ∂J/∂u varies with (u, ρ) — time-varying.

"""
    residual(p, state, design) -> Vector{Float64}

Compute the equilibrium residual R(u, ρ) = A(ρ)u − f.

Lives in V* (same index space as state).  Zero when the state is at
equilibrium.  For the Timoshenko beam this is [∂U/∂w, ∂U/∂ψ], the same
quantity returned as `grad_state` by `energy_gradients`.
"""
function residual(p::AbstractProblem, state, design)
    error("residual not implemented for $(typeof(p))")
end

"""
    obj_state_grad(p, state, design) -> Vector{Float64}

Compute ∂J/∂u — the Fréchet derivative of the objective with respect to the
state, evaluated at (state, design).

Lives in V* (same shape as state).  For compliance J = ⟨f, u⟩ this is the
constant load vector f.  For a general objective this varies with (state,
design) and is time-varying when called along a trajectory.
"""
function obj_state_grad(p::AbstractProblem, state, design)
    error("obj_state_grad not implemented for $(typeof(p))")
end

"""
    apply_stiffness(p, design, v) -> Vector{Float64}

Apply the stiffness operator A(ρ) to an arbitrary vector v (same shape as
state), returning A(ρ)v ∈ V*.

This is the linear operator application *without* the load term f, so
    residual(p, u, ρ) = apply_stiffness(p, ρ, u) − load_vector(p)
but load_vector need not be exposed separately.

Used to compute A(ρ)λ in the forces F_u and F_λ.
"""
function apply_stiffness(p::AbstractProblem, design, v)
    error("apply_stiffness not implemented for $(typeof(p))")
end

"""
    residual_design_grad(p, state, design, lambda) -> Vector{Float64}

Compute (∂R/∂ρ)ᵀ λ — the adjoint-weighted sensitivity of the residual
operator with respect to the design.

Lives in D* (same shape as design).  Combined with ∂J/∂ρ from
`energy_gradients`, this gives the full design force in the 3-player dynamics:

    F_ρ = −∂J/∂ρ − (∂R/∂ρ)ᵀ λ

For linear elasticity R = A(ρ)u − f:
    (∂R/∂ρ)ᵀ λ = (∂A/∂ρ · u)ᵀ λ
which for the Timoshenko beam is the pointwise inner product of the
stiffness-derivative tensors with the strain fields of u and λ.
"""
function residual_design_grad(p::AbstractProblem, state, design, lambda)
    error("residual_design_grad not implemented for $(typeof(p))")
end

# ---------------------------------------------------------------------------
# Optional interface — sensible defaults provided
# ---------------------------------------------------------------------------

"""
    grid_points(p::AbstractProblem) -> Union{Vector, Nothing}

Return the spatial grid (x-coordinates) if the problem is spatially
discretized, or `nothing` for parameter-only problems.
"""
grid_points(p::AbstractProblem) = nothing

"""
    problem_name(p::AbstractProblem) -> String

Human-readable name, used in plot titles and log output.
"""
problem_name(p::AbstractProblem) = "$(typeof(p))"
