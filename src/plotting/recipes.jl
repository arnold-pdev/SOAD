# recipes.jl — Plots.jl recipes for SolverHistory and field profiles
#
# Modeled on orthotrop/local/phaseportrait/methods (bvp_collocation, shooting_exact, timoshenko_flow).
# Uses Plots.jl with multi-panel layouts, θ_ticks for angle axes, and optional save_path.
#
# Loaded only when Plots is available (via Requires.jl).

using Plots

# ---- Shared constants (orthotrop-style) ----

const θ_ticks = (collect(0:π/8:π), ["0", "π/8", "π/4", "3π/8", "π/2", "5π/8", "3π/4", "7π/8", "π"])

"""Break (x, θ) into segments at θ wraps so lines don't cross π boundary when plotting."""
function break_at_wraps(x::AbstractVector, θ::AbstractVector)
    n = length(x)
    n == length(θ) || error("x and θ must have same length")
    n == 0 && (return Float64[], Float64[])
    x_out = Float64[]
    θ_out = Float64[]
    for i in 1:n
        if i > 1 && abs(θ[i] - θ[i-1]) > π/2
            push!(x_out, NaN)
            push!(θ_out, NaN)
        end
        push!(x_out, x[i])
        push!(θ_out, θ[i])
    end
    return x_out, θ_out
end

# ---- Convergence plot (gradient norms vs t, like timoshenko_flow p3) ----

"""
    plot_convergence(hist::SolverHistory; save_path=nothing, yscale=:log10)

Plot gradient norms and objective vs time. Three-panel layout:
  - objective J vs t
  - ‖∇u‖, ‖∇θ‖ vs t (log scale)
  - velocity norms vs t (log scale, if non-zero)

Returns the combined plot. Saves to `save_path` if provided.
"""
function plot_convergence(hist::SolverHistory; save_path=nothing, yscale=:log10)
    t = hist.t
    p1 = plot(xlabel="t", ylabel="J", title="Objective", legend=:best, grid=true)
    plot!(p1, t, hist.objective, label="J", linewidth=2)

    p2 = plot(xlabel="t", ylabel="‖grad‖_L2", title="Gradient norms", legend=:best, grid=true, yscale=yscale)
    plot!(p2, t, hist.grad_state_norm .+ 1e-16, label="‖∇u‖", linewidth=2)
    plot!(p2, t, hist.grad_design_norm .+ 1e-16, label="‖∇θ‖", linewidth=2)

    p3 = plot(xlabel="t", ylabel="‖v‖_L2", title="Velocity norms", legend=:best, grid=true, yscale=yscale)
    plot!(p3, t, hist.vel_state_norm .+ 1e-16, label="‖u̇‖", linewidth=2)
    plot!(p3, t, hist.vel_design_norm .+ 1e-16, label="‖θ̇‖", linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 800), plot_title="Convergence")
    save_path !== nothing && (savefig(p_combined, save_path); println("Saved plot to $save_path"))
    return p_combined
end

"""
    plot_convergence(hists::Vector{<:Tuple}, labels::Vector{String}; save_path=nothing, yscale=:log10)

Overlay multiple histories for comparison (e.g. demo's A, B, C, D).
Each element of hists is (hist, color) or just hist; labels used for legend.
"""
function plot_convergence(hists::AbstractVector, labels::AbstractVector{<:AbstractString}; save_path=nothing, yscale=:log10)
    colors = [:blue, :green, :red, :orange, :purple]
    p1 = plot(xlabel="t", ylabel="J", title="Objective", legend=:best, grid=true)
    p2 = plot(xlabel="t", ylabel="‖∇θ‖_L2", title="Design gradient norm", legend=:best, grid=true, yscale=yscale)
    for (i, hist) in enumerate(hists)
        c = colors[mod1(i, length(colors))]
        lab = i <= length(labels) ? labels[i] : "Run $i"
        plot!(p1, hist.t, hist.objective, label=lab, linewidth=2, color=c)
        plot!(p2, hist.t, hist.grad_design_norm .+ 1e-16, label=lab, linewidth=2, color=c)
    end
    p_combined = plot(p1, p2, layout=(2, 1), size=(800, 600), plot_title="Convergence comparison")
    save_path !== nothing && (savefig(p_combined, save_path); println("Saved plot to $save_path"))
    return p_combined
end

# ---- Solution profiles (θ, w, ψ vs x, like timoshenko_flow p1/p2) ----

"""
    plot_solution_profiles(prob::AbstractProblem, hist::SolverHistory; snapshot_idx=nothing, save_path=nothing)

Plot orientation θ(x) and state fields w(x), ψ(x) for a snapshot from the solver history.

For TimoshenkoProblem: state = [w₁…wₙ, ψ₁…ψₙ], design = [θ₁…θₙ].
Uses `snapshot_idx` (default: last) when `save_fields=true` was used.
If no snapshots, uses current fields from the last recorded state (not available from hist alone —
pass a solver or use plot_design_field with explicit θ).
"""
function plot_solution_profiles(prob::AbstractProblem, hist::SolverHistory; snapshot_idx=nothing, save_path=nothing)
    x = grid_points(prob)
    x === nothing && error("Problem does not provide grid_points")
    n = design_dim(prob)
    idx = snapshot_idx === nothing ? (isempty(hist.design_snapshots) ? 0 : length(hist.design_snapshots)) : snapshot_idx
    if idx <= 0 || idx > length(hist.design_snapshots)
        println("No design snapshots available. Run solver with save_fields=true.")
        return nothing
    end
    θ = hist.design_snapshots[idx]
    state = hist.state_snapshots[idx]
    n_state = state_dim(prob)
    # Timoshenko: state = [w; ψ], n_state = 2n
    n_cell = n
    w = @view state[1:n_cell]
    ψ = if n_state >= 2n_cell; @view state[n_cell+1:2*n_cell]; else; Float64[]; end

    x_θ, θ_θ = break_at_wraps(x, θ)
    p1 = plot(xlabel="x", ylabel="θ(x)", title="Orientation", legend=:none, grid=true, yticks=θ_ticks)
    plot!(p1, x_θ, θ_θ, linewidth=2, color=:blue)

    p2 = plot(xlabel="x", ylabel="w, ψ", title="State", legend=:best, grid=true)
    plot!(p2, x, w, label="w", linewidth=2, color=:blue)
    isempty(ψ) || plot!(p2, x, ψ, label="ψ", linewidth=2, color=:blue, linestyle=:dash)

    p_combined = plot(p1, p2, layout=(2, 1), size=(800, 600), plot_title="$(problem_name(prob)) — snapshot $idx")
    save_path !== nothing && (savefig(p_combined, save_path); println("Saved plot to $save_path"))
    return p_combined
end

"""
    plot_design_field(prob::AbstractProblem, θ::AbstractVector; x=nothing, save_path=nothing)

Plot design field θ(x) along the domain.
"""
function plot_design_field(prob::AbstractProblem, θ::AbstractVector; x=nothing, save_path=nothing)
    x_plot = x === nothing ? grid_points(prob) : x
    x_plot === nothing && error("No grid points available")
    x_θ, θ_θ = break_at_wraps(collect(x_plot), collect(θ))
    p = plot(xlabel="x", ylabel="θ(x)", title="Orientation", legend=:none, grid=true, yticks=θ_ticks)
    plot!(p, x_θ, θ_θ, linewidth=2, color=:blue)
    save_path !== nothing && (savefig(p, save_path); println("Saved plot to $save_path"))
    return p
end

"""
    plot_state_field(prob::AbstractProblem, state::AbstractVector; x=nothing, save_path=nothing)

Plot state fields w(x), ψ(x) for TimoshenkoProblem (state = [w; ψ]).
"""
function plot_state_field(prob::AbstractProblem, state::AbstractVector; x=nothing, save_path=nothing)
    x_plot = x === nothing ? grid_points(prob) : x
    x_plot === nothing && error("No grid points available")
    n = design_dim(prob)
    w = @view state[1:n]
    ψ = if length(state) >= 2n; @view state[n+1:2*n]; else; Float64[]; end
    p = plot(xlabel="x", ylabel="w, ψ", title="State", legend=:best, grid=true)
    plot!(p, x_plot, w, label="w", linewidth=2, color=:blue)
    isempty(ψ) || plot!(p, x_plot, ψ, label="ψ", linewidth=2, color=:blue, linestyle=:dash)
    save_path !== nothing && (savefig(p, save_path); println("Saved plot to $save_path"))
    return p
end
