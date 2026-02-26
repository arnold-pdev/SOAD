# TimoshenkoProblem.jl — Orthotropic Timoshenko cantilever beam
#
# Problem statement
# -----------------
# Find the orientation field θ(x) ∈ [0, π) of an orthotropic elastic beam
# (length L, clamped at x=0, tip load P at x=L) that minimizes the
# regularized compliance
#
#   J(θ) = U(w*(θ), ψ*(θ), θ) - (φ/2) ∫₀ᴸ (θ')² dx
#
# where the potential energy is
#
#   U(w, ψ, θ) = (1/2) ∫₀ᴸ [ B(θ)(ψ')² + S(θ)(ψ - w')² ] dx  -  P·w(L)
#
# and the equilibrium state (w*, ψ*) satisfies the Timoshenko equations
#
#   [S(θ)(ψ - w')]'  = 0       (shear equilibrium)   => V = -P = const
#   [B(θ) ψ']'       = -V = P  (moment equilibrium)
#   w(0) = ψ(0) = 0            (clamped BC)
#
# Stiffness (orthotropic cosine-square model)
# -------------------------------------------
#   B(θ) = B₀ + ΔB·cos²(θ)
#   S(θ) = S₀ + ΔS·cos²(θ - δ)
#
# where δ is a phase offset between the bending and shear anisotropy axes.
#
# Discretization
# --------------
# Staggered finite-difference grid with n cells:
#   cell centers:  xᵢ = (i - 1/2)h,  i = 1,…,n   (w, ψ, θ live here)
#   cell edges:    x_{i+1/2} = i·h,  i = 0,…,n   (fluxes computed here)
#   h = L/n
#
# State vector layout
# -------------------
# state = [w₁, w₂, …, wₙ, ψ₁, ψ₂, …, ψₙ]   (length 2n)
# design = [θ₁, θ₂, …, θₙ]                   (length n)
#
# Gradient sign convention
# ------------------------
# energy_gradients returns (grad_state, grad_design) where:
#   grad_state[1:n]       = ∂U/∂w (positive → decreases w to reduce U)
#   grad_state[n+1:2n]    = ∂U/∂ψ
#   grad_design           = ∂J/∂θ (positive → increases θ; minimizing J
#                                   means moving opposite to grad_design)
#
# The solver is responsible for deciding the sign (ascent vs descent) and
# the timescale of each update.

# ============================================================
#  Parameters
# ============================================================

"""
    TimoshenkoParams

All physical and numerical parameters for the orthotropic Timoshenko beam.

Fields
------
- `L`   : beam length
- `P`   : tip load (negative = downward)
- `B0`  : baseline bending stiffness
- `ΔB`  : bending stiffness anisotropy amplitude
- `S0`  : baseline shear stiffness
- `ΔS`  : shear stiffness anisotropy amplitude
- `δ`   : phase offset between bending and shear anisotropy (radians)
- `φ`   : Tikhonov regularization weight on ∫(θ')²
- `n`   : number of grid cells
"""
struct TimoshenkoParams
    L  ::Float64
    P  ::Float64
    B0 ::Float64
    ΔB ::Float64
    S0 ::Float64
    ΔS ::Float64
    δ  ::Float64
    φ  ::Float64
    n  ::Int
end

"""
    TimoshenkoParams(; kwargs...) -> TimoshenkoParams

Keyword constructor with physically motivated defaults.

Default values correspond to a highly anisotropic material (ΔB, ΔS ≫ B₀, S₀)
for which the orientation problem is non-trivial and multiple local optima
exist.
"""
function TimoshenkoParams(;
    L  = 3.0,
    P  = -5.0,
    B0 = 1.0,
    ΔB = 19.0,
    S0 = 1.0,
    ΔS = 19.0,
    δ  = π/4,
    φ  = 0.001,
    n  = 100
)
    TimoshenkoParams(L, P, B0, ΔB, S0, ΔS, δ, φ, n)
end

# ============================================================
#  Stiffness and its derivatives
# ============================================================

@inline B(θ, p::TimoshenkoParams)  = p.B0 + p.ΔB * cos(θ)^2
@inline S(θ, p::TimoshenkoParams)  = p.S0 + p.ΔS * cos(θ - p.δ)^2

# dB/dθ = -ΔB sin(2θ)
@inline dB(θ, p::TimoshenkoParams) = -p.ΔB * sin(2θ)

# dS/dθ = -ΔS sin(2(θ - δ))
@inline dS(θ, p::TimoshenkoParams) = -p.ΔS * sin(2*(θ - p.δ))

# ============================================================
#  Staggered grid helper
# ============================================================

struct StaggeredGrid
    n    ::Int
    h    ::Float64
    x    ::Vector{Float64}   # cell centers
end

function StaggeredGrid(L::Float64, n::Int)
    h = L / n
    x = [(i - 0.5)*h for i in 1:n]
    StaggeredGrid(n, h, x)
end

# ============================================================
#  Concrete problem type
# ============================================================

"""
    TimoshenkoProblem

Concrete implementation of `AbstractProblem` for the orthotropic Timoshenko
cantilever beam orientation-optimization problem.

Construction
------------
```julia
p = TimoshenkoProblem()                       # default parameters
p = TimoshenkoProblem(TimoshenkoParams(n=200, γ=0.01))
```

State and design layout
-----------------------
- `state`  length 2n: [w₁…wₙ, ψ₁…ψₙ]
- `design` length n:  [θ₁…θₙ]

The equilibrium state for a given θ can be recovered analytically via
`equilibrium_state`, which integrates the Timoshenko equations using the
trapezoidal rule.
"""
struct TimoshenkoProblem <: AbstractProblem
    params ::TimoshenkoParams
    grid   ::StaggeredGrid
end

TimoshenkoProblem() = TimoshenkoProblem(TimoshenkoParams())
TimoshenkoProblem(p::TimoshenkoParams) = TimoshenkoProblem(p, StaggeredGrid(p.L, p.n))

problem_name(::TimoshenkoProblem) = "Timoshenko cantilever (orthotropic)"

# ---- dimensions ------------------------------------------------------------

state_dim(p::TimoshenkoProblem)  = 2 * p.grid.n   # w and ψ concatenated
design_dim(p::TimoshenkoProblem) = p.grid.n

grid_points(p::TimoshenkoProblem) = p.grid.x

# ---- initial conditions ----------------------------------------------------

"""
    initial_state(p::TimoshenkoProblem) -> Vector{Float64}

Zero state (beam unloaded, satisfies the clamped BC trivially).
"""
initial_state(p::TimoshenkoProblem) = zeros(2 * p.grid.n)

"""
    initial_design(p::TimoshenkoProblem) -> Vector{Float64}

Uniform orientation θ ≡ π/4.  This is a non-degenerate starting point that
avoids the symmetry axis of both B and S and lies in the interior of [0, π).
"""
initial_design(p::TimoshenkoProblem) = fill(π/4, p.grid.n)

# ---- boundary conditions ---------------------------------------------------

"""
    enforce_state_bcs!(p, state)

Enforce the clamped (Dirichlet) BC: w(0) = ψ(0) = 0.

On the staggered grid the first cell center is at x₁ = h/2.  The effective
Dirichlet condition is applied by zeroing the first cell value, which is a
standard ghost-cell / half-cell interpretation consistent with the finite-
difference stencil used in `energy_gradients`.
"""
function enforce_state_bcs!(p::TimoshenkoProblem, state)
    state[1]             = 0.0   # w(0) = 0
    state[p.grid.n + 1]  = 0.0   # ψ(0) = 0
    return state
end

"""
    enforce_design_bcs!(p, design)

Wrap orientation angles into [0, π).  Because the material is invariant under
θ → θ + π (no sense in a fiber direction), this keeps the design in a
canonical fundamental domain.
"""
function enforce_design_bcs!(p::TimoshenkoProblem, design)
    @. design = mod(design, π)
    return design
end

# ---- stable timestep -------------------------------------------------------

"""
    stable_dt(p, state, design) -> Float64

Explicit-Euler CFL estimate:

    dt ≤ 0.5 · h² / max(B, S)

This is the standard parabolic CFL for the second-order finite-difference
operator that appears in both the w and ψ equations.
"""
function stable_dt(p::TimoshenkoProblem, state, design)
    h = p.grid.h
    Bmax = maximum(θ -> B(θ, p.params), design)
    Smax = maximum(θ -> S(θ, p.params), design)
    return 0.5 * h^2 / max(Bmax, Smax)
end

# ---- objective -------------------------------------------------------------

"""
    objective(p, state, design) -> Float64

Regularized compliance

    J = U(w, ψ, θ) - (φ/2) ∫ (θ')² dx

where

    U = (1/2) ∫ [ B(θ)(ψ')² + S(θ)(ψ - w')² ] dx  -  P·w(L)

Discrete approximation uses the trapezoidal rule on interior edges.
"""
function objective(p::TimoshenkoProblem, state, design)
    pr = p.params
    g  = p.grid

    w   = @view state[1:g.n]
    ψ   = @view state[g.n+1:2*g.n]
    θ   = design

    U   = _potential_energy(w, ψ, θ, pr, g)
    E_D = _dirichlet_energy(θ, g)
    return U - pr.φ * E_D
end

# ---- energy gradients ------------------------------------------------------

"""
    energy_gradients(p, state, design) -> (grad_state, grad_design)

Compute L²-gradient vectors for the regularized compliance.

Returns
-------
- `grad_state`  :: Vector{Float64} of length 2n
    Concatenation of [∂U/∂w, ∂U/∂ψ].  The solver should subtract this from
    the state velocity (gradient-descent on U).

- `grad_design` :: Vector{Float64} of length n
    ∂J/∂θ.  The solver should *add* this to the design velocity (gradient-
    *ascent* on J — equivalently, descent on -J = compliance).

Derivation sketch (continuous → discrete)
-----------------------------------------
The functional derivatives of U w.r.t. the state fields are:

    δU/δw  =  ∂ₓ[S(θ)(ψ - w')]   (with BC term -P·δ(x-L))
    δU/δψ  = -∂ₓ[B(θ)ψ'] + S(θ)(ψ - w')

The functional derivative of J w.r.t. θ is:

    δJ/δθ = (1/2)[B'(θ)(ψ')² + S'(θ)(ψ - w')²] + φ·θ''

All spatial derivatives are discretized with second-order centered differences
on the staggered grid.  Edge-centered quantities are computed as arithmetic
averages of adjacent cell values.
"""
function energy_gradients(p::TimoshenkoProblem, state, design)
    pr = p.params
    g  = p.grid
    n  = g.n
    h  = g.h

    w  = @view state[1:n]
    ψ  = @view state[n+1:2n]
    θ  = design

    grad_w    = zeros(n)
    grad_ψ    = zeros(n)
    grad_θ    = zeros(n)

    # ----------------------------------------------------------------
    # Loop over interior edges  i = 1, …, n-1
    # Edge i lies between cell i and cell i+1.
    # ----------------------------------------------------------------
    for i in 1:n-1
        # Edge-averaged quantities
        θ_e  = 0.5 * (θ[i] + θ[i+1])
        ψ_e  = 0.5 * (ψ[i] + ψ[i+1])

        # Finite-difference derivatives at edge i
        ψ′_e = (ψ[i+1] - ψ[i]) / h    # ψ' at edge
        w′_e = (w[i+1] - w[i]) / h    # w' at edge

        # Stiffness at edge
        B_e  = B(θ_e, pr)
        S_e  = S(θ_e, pr)
        dB_e = dB(θ_e, pr)
        dS_e = dS(θ_e, pr)

        # Shear strain at edge
        φ_e  = ψ_e - w′_e

        # ---- grad_w : ∂U/∂w from S(θ)φ flux divergence ----
        # Contribution of shear flux across edge i to adjacent cells:
        #   ∂U/∂w[i]   += +S_e · φ_e  (outgoing from cell i)
        #   ∂U/∂w[i+1] -= +S_e · φ_e  (incoming to cell i+1)
        flux_w          = S_e * φ_e
        grad_w[i]      += flux_w
        grad_w[i+1]    -= flux_w

        # ---- grad_ψ : ∂U/∂ψ from bending + shear ----
        # Bending flux  B_e·ψ'_e  contributes ±1/h per adjacent cell.
        # Shear  S_e·φ_e  is split equally (h/2 weight per cell).
        bending_flux    = B_e * ψ′_e
        grad_ψ[i]      += (-bending_flux + 0.5*h * S_e * φ_e) / h * h
        grad_ψ[i+1]    += ( bending_flux + 0.5*h * S_e * φ_e) / h * h
        # Simplify: factor of h/h = 1 from the flux; 0.5*h from the cell weight
        # Rewrite cleanly:
        grad_ψ[i]      = grad_ψ[i]   # accumulated above; see below for the clean pass

        # ---- grad_θ : ∂J/∂θ from stiffness anisotropy ----
        # δJ/δθ (edge contribution) = (h/2)[dB·(ψ'²) + dS·φ²]  per edge
        # Split equally between the two adjacent cell centers.
        dE_dθ_edge = 0.5*h * (dB_e * ψ′_e^2 + dS_e * φ_e^2)
        grad_θ[i]      += 0.5 * dE_dθ_edge
        grad_θ[i+1]    += 0.5 * dE_dθ_edge
    end

    # Redo the grad_ψ accumulation with a clean two-pass approach
    # to avoid the messy in-place accumulation above.
    grad_ψ .= 0.0
    for i in 1:n-1
        θ_e  = 0.5 * (θ[i] + θ[i+1])
        ψ_e  = 0.5 * (ψ[i] + ψ[i+1])
        ψ′_e = (ψ[i+1] - ψ[i]) / h
        w′_e = (w[i+1] - w[i]) / h
        B_e  = B(θ_e, pr)
        S_e  = S(θ_e, pr)
        φ_e  = ψ_e - w′_e

        # Divergence of bending flux: +B_e·ψ'_e into cell i+1, −into cell i
        grad_ψ[i]   -= B_e * ψ′_e     # -∂ₓ(Bψ') contribution at cell i
        grad_ψ[i+1] += B_e * ψ′_e     # ∂ₓ(Bψ') contribution at cell i+1

        # Shear term: S_e·φ_e distributed equally to both cells
        grad_ψ[i]   += 0.5 * h * S_e * φ_e
        grad_ψ[i+1] += 0.5 * h * S_e * φ_e
    end
    # Normalize: divide by h to get the L²-gradient (not integrated)
    grad_ψ ./= h

    # ---- Tip load contribution to grad_w ----
    # External work W = P·w(L) contributes -P to ∂U/∂w at the last cell.
    grad_w[n] -= pr.P

    # ---- θ regularization: +φ·θ'' ----
    # We want ∂J/∂θ = ∂U/∂θ + φ·θ'' (Tikhonov).
    # Discrete Laplacian with zero-flux (Neumann) BCs on θ:
    grad_θ .+= _theta_laplacian(θ, pr.φ, h)

    grad_state  = vcat(grad_w, grad_ψ)
    grad_design = grad_θ
    return grad_state, grad_design
end

# ---- equilibrium solve -----------------------------------------------------

"""
    equilibrium_state(p, design) -> Vector{Float64}

Compute the Timoshenko equilibrium state u*(θ) analytically for a given
design field θ.

For a cantilever with tip shear load P and no distributed loads, the
equilibrium can be integrated directly:

    ψ'(x) = P(L - x) / B(θ(x))     ⟹  ψ via trapezoidal integration
    w'(x) = ψ(x) + P / S(θ(x))     ⟹  w via trapezoidal integration

with BCs ψ(0) = w(0) = 0.

This exact (no Newton solve needed) equilibrium recovery is specific to the
cantilever geometry with a single tip load.  For other BCs or distributed
loads the function would need to be replaced with a linear system solve.
"""
function equilibrium_state(p::TimoshenkoProblem, design)
    pr = p.params
    g  = p.grid
    n  = g.n
    h  = g.h
    x  = g.x
    θ  = design

    w  = zeros(n)
    ψ  = zeros(n)

    # Precompute dψ/dx = P(L - x) / B(θ) at each cell center
    dψdx = [pr.P * (pr.L - x[i]) / B(θ[i], pr) for i in 1:n]

    # Integrate from x=0 (cell 1 is at h/2; ghost cell at 0 has ψ=w=0)
    # Use trapezoidal rule between consecutive cell centers.
    # For cell 1: integrate from 0 to x₁ ≈ h/2 using midpoint rule
    ψ[1] = dψdx[1] * x[1]
    w[1] = (pr.P / S(θ[1], pr)) * x[1]   # ψ(0)=0, so dw/dx ≈ P/S at left

    for i in 2:n
        Δx    = x[i] - x[i-1]
        # ψ: trapezoidal rule
        ψ[i]  = ψ[i-1] + 0.5 * Δx * (dψdx[i-1] + pr.P * (pr.L - x[i]) / B(θ[i], pr))
        # w: trapezoidal rule — dw/dx = ψ + P/S(θ)
        dw_prev = ψ[i-1] + pr.P / S(θ[i-1], pr)
        dw_curr = ψ[i]   + pr.P / S(θ[i],   pr)
        w[i]  = w[i-1] + 0.5 * Δx * (dw_prev + dw_curr)
    end

    return vcat(w, ψ)
end

# ============================================================
#  General (3-player λ) interface
# ============================================================

"""
    residual(p::TimoshenkoProblem, state, design) -> Vector{Float64}

Equilibrium residual R(u, θ) = A(θ)u − f for the Timoshenko beam.

For compliance, this equals the `grad_state` returned by `energy_gradients`
(they are the same computation).  Exposed separately so the λ-solver can
call it without also computing grad_design.
"""
function residual(p::TimoshenkoProblem, state, design)
    grad_state, _ = energy_gradients(p, state, design)
    return grad_state
end

"""
    obj_state_grad(p::TimoshenkoProblem, state, design) -> Vector{Float64}

∂J/∂u for the compliance objective J = ⟨f, u⟩ = −P·w(L).

The functional derivative is the load vector: δJ/δw = −P·δ(x−L) at the tip,
δJ/δψ = 0 everywhere.  In the discrete setting this is a vector of zeros
except for index n (last w-cell), which holds −P.

This is constant (independent of state and design) for compliance, but the
interface is defined for general time-varying objectives.
"""
function obj_state_grad(p::TimoshenkoProblem, state, design)
    n   = p.grid.n
    g   = zeros(2n)
    g[n] = -p.params.P   # δJ/δw at tip cell; δJ/δψ = 0
    return g
end

"""
    apply_stiffness(p::TimoshenkoProblem, design, v) -> Vector{Float64}

Apply the Timoshenko stiffness operator A(θ) to an arbitrary vector
v = [v_w; v_ψ] (same shape as state), returning A(θ)v ∈ V*.

This runs the same bending+shear stencil as `energy_gradients` but with v
substituted for the state and **without** subtracting the tip load f.
Concretely:

    residual(p, u, θ) = apply_stiffness(p, θ, u) − obj_state_grad(p, u, θ)

since for compliance ∂J/∂u = f = load, and A(θ)u − f = R(u,θ).

Used in the λ-solver to compute A(θ)λ for the forces F_u and F_λ.
"""
function apply_stiffness(p::TimoshenkoProblem, design, v)
    pr = p.params
    g  = p.grid
    n  = g.n
    h  = g.h
    θ  = design

    v_w = @view v[1:n]
    v_ψ = @view v[n+1:2n]

    Av_w = zeros(n)
    Av_ψ = zeros(n)

    for i in 1:n-1
        θ_e   = 0.5 * (θ[i] + θ[i+1])
        vψ_e  = 0.5 * (v_ψ[i] + v_ψ[i+1])
        vψ′_e = (v_ψ[i+1] - v_ψ[i]) / h
        vw′_e = (v_w[i+1] - v_w[i]) / h
        B_e   = B(θ_e, pr)
        S_e   = S(θ_e, pr)
        γ_e   = vψ_e - vw′_e   # shear strain of v

        # Shear flux contribution to Av_w (divergence of S·γ)
        flux_w       = S_e * γ_e
        Av_w[i]     += flux_w
        Av_w[i+1]   -= flux_w
    end

    # Bending + shear contributions to Av_ψ (two-pass, clean accumulation)
    for i in 1:n-1
        θ_e   = 0.5 * (θ[i] + θ[i+1])
        vψ_e  = 0.5 * (v_ψ[i] + v_ψ[i+1])
        vψ′_e = (v_ψ[i+1] - v_ψ[i]) / h
        vw′_e = (v_w[i+1] - v_w[i]) / h
        B_e   = B(θ_e, pr)
        S_e   = S(θ_e, pr)
        γ_e   = vψ_e - vw′_e

        # Divergence of bending flux B·vψ'
        Av_ψ[i]   -= B_e * vψ′_e
        Av_ψ[i+1] += B_e * vψ′_e

        # Shear term distributed equally to adjacent cells
        Av_ψ[i]   += 0.5 * h * S_e * γ_e
        Av_ψ[i+1] += 0.5 * h * S_e * γ_e
    end
    Av_ψ ./= h

    return vcat(Av_w, Av_ψ)
end

"""
    residual_design_grad(p::TimoshenkoProblem, state, design, lambda) -> Vector{Float64}

Compute (∂R/∂θ)ᵀ λ — the adjoint-weighted sensitivity of the residual with
respect to the design θ.

For the Timoshenko beam with R(u,θ) = A(θ)u − f:

    (∂R/∂θ)ᵀ λ = (∂A/∂θ · u)ᵀ λ

The bilinear form ⟨λ, (∂A/∂θ[i])u⟩ evaluated at cell i gives a scalar
sensitivity.  Using the edge-averaged stiffness derivatives:

    δ⟨λ, Au⟩/δθ_i = (1/2) h · [dB(θ_e) · λψ'·uψ' + dS(θ_e) · λγ·uγ]

summed over adjacent edges, where uγ = uψ − uw' and λγ = λψ − λw'.

This is the same stencil as the design gradient in `energy_gradients`, but
with u and λ in the two "slots" of the bilinear form instead of u in both.
"""
function residual_design_grad(p::TimoshenkoProblem, state, design, lambda)
    pr = p.params
    g  = p.grid
    n  = g.n
    h  = g.h
    θ  = design

    u_w = @view state[1:n]
    u_ψ = @view state[n+1:2n]
    λ_w = @view lambda[1:n]
    λ_ψ = @view lambda[n+1:2n]

    sens = zeros(n)

    for i in 1:n-1
        θ_e   = 0.5 * (θ[i] + θ[i+1])
        dB_e  = dB(θ_e, pr)
        dS_e  = dS(θ_e, pr)

        # State strain fields at edge i
        uψ′_e = (u_ψ[i+1] - u_ψ[i]) / h
        uw′_e = (u_w[i+1] - u_w[i]) / h
        uψ_e  = 0.5 * (u_ψ[i] + u_ψ[i+1])
        uγ_e  = uψ_e - uw′_e

        # Adjoint strain fields at edge i
        λψ′_e = (λ_ψ[i+1] - λ_ψ[i]) / h
        λw′_e = (λ_w[i+1] - λ_w[i]) / h
        λψ_e  = 0.5 * (λ_ψ[i] + λ_ψ[i+1])
        λγ_e  = λψ_e - λw′_e

        # Adjoint-weighted sensitivity: ⟨λ, dA/dθ · u⟩ at this edge
        dsens_edge = 0.5 * h * (dB_e * λψ′_e * uψ′_e + dS_e * λγ_e * uγ_e)

        # Split equally to adjacent cell centers
        sens[i]   += 0.5 * dsens_edge
        sens[i+1] += 0.5 * dsens_edge
    end

    return sens
end

# ============================================================
#  Private helpers
# ============================================================

# Discrete potential energy  U = E_bending + E_shear - W_tip
function _potential_energy(w, ψ, θ, pr, g)
    n = g.n
    h = g.h
    U = 0.0
    for i in 1:n-1
        θ_e  = 0.5 * (θ[i] + θ[i+1])
        ψ_e  = 0.5 * (ψ[i] + ψ[i+1])
        ψ′_e = (ψ[i+1] - ψ[i]) / h
        w′_e = (w[i+1] - w[i]) / h
        φ_e  = ψ_e - w′_e
        U   += 0.5 * h * (B(θ_e, pr) * ψ′_e^2 + S(θ_e, pr) * φ_e^2)
    end
    U -= pr.P * w[n]   # tip work W = P·w(L)
    return U
end

# Discrete Dirichlet energy  (1/2) ∫ (θ')² dx — NOT pre-multiplied by φ
function _dirichlet_energy(θ, g)
    n = g.n
    h = g.h
    E = 0.0
    for i in 1:n-1
        θ′ = (θ[i+1] - θ[i]) / h
        E += 0.5 * h * θ′^2
    end
    return E
end

# Discrete Laplacian for θ with zero-flux (Neumann) BCs.
# Returns φ·θ'' as a vector of length n, for adding to grad_θ.
function _theta_laplacian(θ, φ, h)
    n   = length(θ)
    lap = zeros(n)
    for i in 2:n-1
        lap[i] = φ * (θ[i-1] - 2θ[i] + θ[i+1]) / h^2
    end
    # Neumann BC: one-sided differences at boundaries
    # θ'(0) = 0  ⟹  θ[0] = θ[1]  (ghost cell)
    lap[1] = φ * (θ[2]   - θ[1])   / h^2   # second difference with ghost θ[0] = θ[1]
    lap[n] = φ * (θ[n-1] - θ[n])   / h^2   # second difference with ghost θ[n+1] = θ[n]
    return lap
end
