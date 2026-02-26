# SOAD.jl — Package entry point
#
# Re-exports AutonomousDynamics for use as `using SOAD`.

module SOAD

include("AutonomousDynamics.jl")
using .AutonomousDynamics

# Re-export everything from AutonomousDynamics
export AbstractProblem, TimoshenkoProblem, TimoshenkoParams
export state_dim, design_dim, initial_state, initial_design
export energy_gradients, equilibrium_state, objective
export enforce_state_bcs!, enforce_design_bcs!, stable_dt
export grid_points, problem_name
export AbstractSchedule, ScheduleParams
export ConstantSchedule, NesterovSchedule, FIRESchedule
export step!, reset!
export AbstractIntegrator, SymplecticEuler, RK4Integrator
export integrate!
export residual, obj_state_grad, apply_stiffness, residual_design_grad
export AbstractSolver, SolverHistory
export GradientFlowSolver, InertialSolver
export LambdaSolver, current_lambda, FixedRatio, AdaptiveTol
export solve!, current_state, current_design
export plot_convergence, plot_convergence_unicode
export plot_solution_profiles, plot_design_field, plot_state_field
export break_at_wraps

end # module SOAD
