# unicode.jl — UnicodePlots terminal fallback for convergence
#
# Provides plot_convergence_unicode for terminal output when Plots.jl is not available.
# Loaded only when UnicodePlots is available (via Requires.jl).

using UnicodePlots

"""
    plot_convergence_unicode(hist::SolverHistory)

Terminal convergence plot of log₁₀‖∇θ‖ vs t. Returns the plot object (for display/println).
"""
function plot_convergence_unicode(hist::SolverHistory)
    lineplot(hist.t, log10.(hist.grad_design_norm .+ 1e-16);
             title="log₁₀‖∇θ‖ vs t", xlabel="t", ylabel="log₁₀‖∇θ‖")
end

"""
    plot_convergence_unicode(hists::AbstractVector, labels::AbstractVector{<:AbstractString})

Overlay multiple histories for comparison. UnicodePlots has limited multi-series support;
this plots the first history and prints a note about the others.
"""
function plot_convergence_unicode(hists::AbstractVector, labels::AbstractVector{<:AbstractString})
    p = lineplot(hists[1].t, log10.(hists[1].grad_design_norm .+ 1e-16);
                 title="log₁₀‖∇θ‖ vs t", name=labels[1], xlabel="t", ylabel="log₁₀‖∇θ‖")
    for i in 2:min(length(hists), length(labels))
        lineplot!(p, hists[i].t, log10.(hists[i].grad_design_norm .+ 1e-16); name=labels[i])
    end
    return p
end
