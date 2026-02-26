# AutonomousDynamics.jl — Main module entry point
#
# Framework for inertial / dynamic-relaxation optimization with independent
# schedules for state and design variables.
#
# Architecture overview
# ---------------------
#
#   AbstractProblem         (src/problems/Problem.jl)
#   └── TimoshenkoProblem   (src/problems/TimoshenkoProblem.jl)
#       • state  = [w₁…wₙ, ψ₁…ψₙ]  (transverse displacement + rotation)
#       • design = [θ₁…θₙ]          (material orientation field)
#
#   AbstractSchedule        (src/schedules/Schedule.jl)
#   ├── ConstantSchedule    — fixed (mass, damping); covers 1st-order and heavy-ball
#   ├── NesterovSchedule    — α(t) = r/t; continuous-time Nesterov (SBC 2016)
#   └── FIRESchedule        — adaptive FIRE with velocity-restart signal
#
#   AbstractSolver          (src/solvers/Solver.jl)
#   ├── GradientFlowSolver  — first-order (m=0); optionally equilibrium-constrained
#   └── InertialSolver      — second-order (m>0); same constrained option
#
# Solver modes
# ------------
# Both solvers support two modes selected via `constrained`:
#
#   constrained = false  (joint dynamics, default)
#       State and design evolve simultaneously.  The state can depart from
#       equilibrium.  This directly simulates the two-timescale coupled system.
#
#   constrained = true  (equilibrium-projected, slow-fast limit)
#       After each design step the state is reset to u*(θ) via an exact
#       equilibrium solve.  Numerically more stable; theoretically corresponds
#       to the ε→0 limit of the state/design timescale ratio.
#
# Quick start
# -----------
# ```julia
# include("src/AutonomousDynamics.jl")
# using .AutonomousDynamics
#
# prob = TimoshenkoProblem()
#
# # --- First-order gradient flow, joint dynamics ---
# solver = GradientFlowSolver(
#     prob,
#     ConstantSchedule(),            # state  schedule
#     ConstantSchedule(damping=0.5); # design schedule (larger step)
#     T=5.0, stride=20, save_fields=true
# )
# hist = solve!(solver)
#
# # --- Nesterov-accelerated, equilibrium-constrained design only ---
# solver2 = InertialSolver(
#     prob,
#     ConstantSchedule(),   # state schedule (unused in constrained mode)
#     NesterovSchedule();
#     constrained=true, T=5.0
# )
# hist2 = solve!(solver2)
#
# # --- FIRE, joint ---
# solver3 = InertialSolver(
#     prob,
#     FIRESchedule(),
#     FIRESchedule();
#     T=5.0
# )
# hist3 = solve!(solver3)
# ```
#
# Adding a new problem
# --------------------
# 1. Create src/problems/MyProblem.jl with a struct <: AbstractProblem.
# 2. Implement all methods from the AbstractProblem interface.
# 3. Add `include("problems/MyProblem.jl")` and export your type below.
#    No solver code needs to change.
#
# Adding a new schedule
# ---------------------
# 1. Create src/schedules/MySchedule.jl with a struct <: AbstractSchedule.
# 2. Implement `step!(s::MySchedule, k, t, grad, vel) -> ScheduleParams`.
# 3. Add include + export below.

module AutonomousDynamics

using Printf
using LinearAlgebra
using Requires

# ---- Problems --------------------------------------------------------------
include("problems/Problem.jl")
include("problems/TimoshenkoProblem.jl")

export AbstractProblem
export TimoshenkoProblem, TimoshenkoParams
export state_dim, design_dim, initial_state, initial_design
export energy_gradients, equilibrium_state, objective
export enforce_state_bcs!, enforce_design_bcs!, stable_dt
export grid_points, problem_name
export residual, obj_state_grad, apply_stiffness, residual_design_grad

# ---- Schedules -------------------------------------------------------------
include("schedules/Schedule.jl")
include("schedules/ConstantSchedule.jl")
include("schedules/NesterovSchedule.jl")
include("schedules/FIRESchedule.jl")

export AbstractSchedule, ScheduleParams
export ConstantSchedule, NesterovSchedule, FIRESchedule
export step!, reset!

# ---- Integrators -----------------------------------------------------------
include("integrators/Integrator.jl")
include("integrators/SymplecticEuler.jl")
include("integrators/RK4.jl")

export AbstractIntegrator, SymplecticEuler, RK4Integrator
export integrate!

# ---- Solvers — shared infrastructure ---------------------------------------
include("solvers/Solver.jl")

export AbstractSolver, SolverHistory
export solve!, current_state, current_design

# ---- Solvers — compliance (2-player) ---------------------------------------
include("solvers/compliance/GradientFlowSolver.jl")
include("solvers/compliance/InertialSolver.jl")

export GradientFlowSolver, InertialSolver

# ---- Solvers — general (3-player λ) ----------------------------------------
include("solvers/general/LambdaSolver.jl")

export LambdaSolver, current_lambda, FixedRatio, AdaptiveTol

# ---- Plotting (optional, via Requires.jl) -----------------------------------
export plot_convergence, plot_convergence_unicode
export plot_solution_profiles, plot_design_field, plot_state_field
export break_at_wraps

# Stubs (overwritten when Plots/UnicodePlots load)
function plot_convergence(hist; kwargs...) error("Load Plots for GUI plotting: using Plots") end
function plot_convergence_unicode(hist) error("Load UnicodePlots for terminal plotting: using UnicodePlots") end
function plot_solution_profiles(prob, hist; kwargs...) error("Load Plots: using Plots") end
function plot_design_field(prob, θ; kwargs...) error("Load Plots: using Plots") end
function plot_state_field(prob, state; kwargs...) error("Load Plots: using Plots") end
break_at_wraps(x, θ) = error("break_at_wraps is defined in plotting/recipes.jl; load Plots: using Plots")

function __init__()
    @require UnicodePlots = "b8865323-c70e-4414-a5af-42eaca4e2ef0" include("plotting/unicode.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plotting/recipes.jl")
end

end # module AutonomousDynamics
