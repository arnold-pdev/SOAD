# TOAD.jl — Package entry point
#
# Re-exports AdaptiveDynamics for use as `using TOAD`.

module TOAD

include("AdaptiveDynamics.jl")
using .AdaptiveDynamics

# Re-export everything from AdaptiveDynamics
export AbstractProblem, TimoshenkoProblem, TimoshenkoParams
export state_dim, design_dim, initial_state, initial_design
export energy_gradients, equilibrium_state, objective
export enforce_state_bcs!, enforce_design_bcs!, stable_dt
export grid_points, problem_name
export AbstractSchedule, ScheduleParams
export ConstantSchedule, NesterovSchedule, FIRESchedule
export step!, reset!
export AbstractSolver, SolverHistory
export GradientFlowSolver, InertialSolver
export solve!, current_state, current_design
export plot_convergence, plot_convergence_unicode
export plot_solution_profiles, plot_design_field, plot_state_field
export break_at_wraps

end # module TOAD
