# TOAD

A Julia framework for **inertial / dynamic-relaxation optimisation** with independent schedules for state and design variables. Built for structural optimisation problems where a fast state variable (e.g. displacement) is coupled to a slow design variable (e.g. material orientation).

## Features

- **Two solver types**: first-order gradient flow and second-order inertial dynamics
- **Flexible schedules**: constant (mass/damping), Nesterov (α = r/t), and FIRE adaptive relaxation
- **Two dynamics modes**:
  - **Joint dynamics** — state and design evolve simultaneously
  - **Equilibrium-constrained** — state projected onto equilibrium after each design step (slow-fast limit)
- **Extensible design**: plug in new problems and schedules without modifying solver code

## Quick Start

```julia
include("src/AdaptiveDynamics.jl")
using .AdaptiveDynamics

prob = TimoshenkoProblem()

# First-order gradient flow, joint dynamics
solver = GradientFlowSolver(
    prob,
    ConstantSchedule(),            # state schedule
    ConstantSchedule(damping=0.5); # design schedule
    T=5.0, stride=20, save_fields=true
)
hist = solve!(solver)

# Nesterov-accelerated, equilibrium-constrained
solver2 = InertialSolver(
    prob,
    ConstantSchedule(),
    NesterovSchedule();
    constrained=true, T=5.0
)
hist2 = solve!(solver2)
```

## Running the Demo

The demo compares four solver configurations on the Timoshenko beam problem:

```bash
julia --project=. examples/demo_timoshenko.jl
```

Configurations:
- **(A)** First-order gradient flow, joint dynamics
- **(B)** First-order gradient flow, equilibrium-constrained
- **(C)** Nesterov-accelerated inertial, constrained
- **(D)** FIRE adaptive inertial, joint

Install `Plots` or `UnicodePlots` for convergence plots.

## Project Structure

```
TOAD/
├── src/
│   ├── AdaptiveDynamics.jl    # Main module
│   ├── problems/
│   │   ├── Problem.jl         # Abstract interface
│   │   └── TimoshenkoProblem.jl
│   ├── schedules/
│   │   ├── Schedule.jl
│   │   ├── ConstantSchedule.jl
│   │   ├── NesterovSchedule.jl
│   │   └── FIRESchedule.jl
│   └── solvers/
│       ├── Solver.jl
│       ├── FirstOrderSolver.jl
│       └── SecondOrderSolver.jl
└── examples/
    └── demo_timoshenko.jl
```

## Architecture

| Component | Description |
|-----------|-------------|
| **AbstractProblem** | Interface for optimisation problems (state + design variables) |
| **AbstractSchedule** | Provides (mass, damping) or adaptive params per step |
| **AbstractSolver** | Integrates inertial dynamics using problem + schedules |

The continuous equations:

- **State**: m_u · ü + α_u · u̇ = −∂U/∂u  
- **Design**: m_θ · θ̈ + α_θ · θ̇ = +∂J/∂θ  

Special cases: m=0 → gradient flow; m=1, α=3/t → Nesterov; m=1, α=FIRE → FIRE relaxation.

## Extending the Framework

**Adding a new problem**: Create a struct `<: AbstractProblem` and implement the interface (`state_dim`, `design_dim`, `energy_gradients`, `equilibrium_state`, etc.). See `src/problems/Problem.jl` for the full interface.

**Adding a new schedule**: Create a struct `<: AbstractSchedule` and implement `step!(s, k, t, grad, vel) -> ScheduleParams`.

## Plotting

Plotting is optional (loaded via Requires.jl when Plots or UnicodePlots is available):

```julia
using Plots  # or using UnicodePlots for terminal plots
plot_convergence(hist)                                    # single history
plot_convergence([histA, histB], ["A", "B"]; save_path="out.png")  # comparison
plot_solution_profiles(prob, hist)                        # θ(x), w(x), ψ(x) (last snapshot)
plot_design_field(prob, θ)                                # θ(x) only
plot_state_field(prob, state)                             # w(x), ψ(x)
```

Style follows orthotrop phase-portrait methods: multi-panel layouts, θ ticks, `break_at_wraps` for angle wrapping.

## Requirements

- Julia 1.x
- Standard library: `Printf`, `LinearAlgebra`
- Optional: `Plots` (GUI) or `UnicodePlots` (terminal) for plotting
