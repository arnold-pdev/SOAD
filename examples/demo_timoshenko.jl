# demo_timoshenko.jl — Demonstration of AdaptiveDynamics on the Timoshenko beam
#
# This script runs four solver configurations on the same problem and plots
# the convergence history so they can be compared side by side:
#
#   (A) First-order gradient flow,        joint      (baseline)
#   (B) First-order gradient flow,        constrained (slow-fast limit)
#   (C) Nesterov-accelerated inertial,    constrained
#   (D) FIRE adaptive inertial,           joint
#
# The comparison (A) vs (B) isolates the effect of exact equilibrium projection.
# The comparison (B) vs (C) isolates the acceleration effect at fixed mode.
# The comparison (C) vs (D) compares two adaptive schedules.
#
# Run from the adaptive_dynamics directory:
#   julia --project=. examples/demo_timoshenko.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "AdaptiveDynamics.jl"))
using .AdaptiveDynamics
using Printf

# ============================================================
#  Problem setup
# ============================================================

params = TimoshenkoParams(
    L  = 3.0,
    P  = -5.0,
    B0 = 1.0,  ΔB = 19.0,
    S0 = 1.0,  ΔS = 19.0,
    δ  = π/4,
    γ  = 0.001,
    n  = 100
)
prob = TimoshenkoProblem(params)

println("Problem: ", problem_name(prob))
println("  n cells = ", params.n, "  L = ", params.L, "  P = ", params.P)
println("  γ = ", params.γ)
println()

# ============================================================
#  Solver settings (shared)
# ============================================================

T       = 8.0     # total simulation time
stride  = 50      # record every 50 steps
verbose = true

# dt: use 80% of the CFL limit for safety
dt = 0.8 * stable_dt(prob, initial_state(prob), initial_design(prob))
@printf("CFL-safe dt = %.3e  (using dt = %.3e)\n\n", dt/0.8, dt)

# ============================================================
#  (A) First-order joint dynamics
# ============================================================

println("="^60)
println("(A) GradientFlowSolver — joint dynamics")
println("="^60)

solverA = GradientFlowSolver(
    prob,
    ConstantSchedule(damping=1.0),   # state
    ConstantSchedule(damping=1.0);   # design
    dt=dt, T=T, stride=stride, save_fields=true,
    constrained=false, verbose=verbose
)
histA = solve!(solverA)
println()

# ============================================================
#  (B) First-order constrained (equilibrium-projected)
# ============================================================

println("="^60)
println("(B) GradientFlowSolver — equilibrium-constrained")
println("="^60)

solverB = GradientFlowSolver(
    prob,
    ConstantSchedule(damping=1.0),   # state (unused in constrained mode)
    ConstantSchedule(damping=1.0);   # design
    dt=dt, T=T, stride=stride, save_fields=true,
    constrained=true, verbose=verbose
)
histB = solve!(solverB)
println()

# ============================================================
#  (C) Nesterov inertial, constrained
# ============================================================

println("="^60)
println("(C) InertialSolver (Nesterov) — equilibrium-constrained")
println("="^60)

solverC = InertialSolver(
    prob,
    ConstantSchedule(damping=1.0),   # state (unused in constrained mode)
    NesterovSchedule(r=3.0);         # design: SBC continuous Nesterov
    dt=dt, T=T, stride=stride, save_fields=true,
    constrained=true, verbose=verbose
)
histC = solve!(solverC)
println()

# ============================================================
#  (D) FIRE inertial, joint
# ============================================================

println("="^60)
println("(D) InertialSolver (FIRE) — joint dynamics")
println("="^60)

solverD = InertialSolver(
    prob,
    FIRESchedule(α_start=0.1, f_α=0.99, N_min=5),   # state
    FIRESchedule(α_start=0.1, f_α=0.99, N_min=5);   # design
    dt=dt, T=T, stride=stride, save_fields=true,
    constrained=false, verbose=verbose
)
histD = solve!(solverD)
println()

# ============================================================
#  Summary table
# ============================================================

println("="^60)
println("Summary: final objective values")
println("="^60)
for (label, hist) in [("(A) 1st-order joint", histA),
                       ("(B) 1st-order constrained", histB),
                       ("(C) Nesterov constrained", histC),
                       ("(D) FIRE joint", histD)]
    J_final  = isempty(hist.objective) ? NaN : hist.objective[end]
    gd_final = isempty(hist.grad_design_norm) ? NaN : hist.grad_design_norm[end]
    @printf("  %-28s  J = %10.6f   |∇θ| = %.3e\n", label, J_final, gd_final)
end
println()

# ============================================================
#  Final orientation profiles
# ============================================================

x = grid_points(prob)
println("Final θ profiles (first 10 cells):")
for (label, solver) in [("(A)", solverA), ("(B)", solverB),
                          ("(C)", solverC), ("(D)", solverD)]
    θ = current_design(solver)
    @printf("  %s  θ ∈ [%.4f, %.4f]  mean = %.4f\n",
            label, minimum(θ), maximum(θ), sum(θ)/length(θ))
end

# ============================================================
#  Plotting (Plots.jl or UnicodePlots)
# ============================================================

hists = [histA, histB, histC, histD]
labels = ["(A) 1st joint", "(B) 1st constrained", "(C) Nesterov", "(D) FIRE"]

if Base.find_package("Plots") !== nothing
    using Plots
    p = plot_convergence(hists, labels; save_path=joinpath(@__DIR__, "..", "demo_convergence.png"))
    display(p)
elseif Base.find_package("UnicodePlots") !== nothing
    using UnicodePlots
    println(plot_convergence_unicode(hists, labels))
else
    println("(Install Plots or UnicodePlots for convergence plots.)")
    println()
    println("Convergence data available in histA, histB, histC, histD:")
    println("  fields: .t, .objective, .grad_design_norm, .grad_state_norm")
    println("  field snapshots (if save_fields=true): .design_snapshots, .state_snapshots")
end
