#using Distributed
#addprocs(2)

##

begin
    using Pkg
    Pkg.activate(@__DIR__)

    using ProbabilisticStability

    using OrdinaryDiffEq
    #using DiffEqCallbacks
    #using DynamicalSystems
    using MCBB
    #using Distributions, Clustering
    using Distances
    using QuasiMonteCarlo # to compare different samplings
    using Measurements # nice uncertainty handling
    using NetworkDynamics
    using LightGraphs
    #using NLsolve
    #using StatsBase
    using DataFrames

    using Random
    Random.seed!(42);

    using BenchmarkTools
    using Plots
    using StatsPlots
    theme(:solarized_light)
    using LaTeXStrings
end

##

"""
Here, we want to reproduce step-by-step Fig. 1 from the publication:

Menck,P & Kurths, J.
Topological Identification of Weak Points in Power Grids.
Nonlinear Dynamics of Electronic Systems, 144-147, 2012
"""

struct SwingPars <: DEParameters
    P
    α
    K
    θgrid
end

function swing!(du, u, p, t)ms=1.
    n = length(u) ÷ 2
    ω = @view u[1:n]
    θ = @view u[n+1:2n]
    dω = @view du[1:n]
    dθ = @view du[n+1:2n]
    @. dω = p.P - p.α * ω - p.K .* sin.(θ - p.θgrid)
    @. dθ = ω
end




## compare MonteCarlo and QuasiMonteCarlo sampling, parallel_integrator and EnsembleProblem

begin
    T = 500

    # define sampling region
    lb = [-100.0, -π]
    ub = -lb

    sobol_estimate = []
    #uniform_estimate = []
    αs = 0.:0.01:0.4
end

p = SwingPars(1.0, 0.1, 8.0, 0.0)
fp = [0.0, asin(p.P / p.K)]

# ode = ODEProblem(swing!, fp .+ rand().*ub,  (0., 500.), p)
# sol = solve(ode, Rosenbrock23())
#     t=1e-5
#     plot(sol, vars=((t,x)-> (t, abs(x)), 0, 1), yscale=:log10)#, ylims=(0,t))
#     plot!(sol.t, [evaluate(PeriodicEuclidean([Inf,2π]), fp, u) for u in sol.u])
#     plot!(sol.t, [abs(u[1]) for u in sol.u])
#     #plot!(sol.t, ode.u0[1] .* exp.(-sol.t * p.α))
#     vline!([- log(t / ode.u0[1]) / p.α, min(1000, 10*log(10)/p.α)])

##

ode = ODEProblem(swing!, fp .+ 0.1rand(length(fp)), (0.0, 100.), p)
sol = solve(ode, Rosenbrock23())

# custom indicator function
# could provide defaults? Rectangle with bound for each dimension
function set_indicator(p)
    return -0.2 < p[1] < 0.2
end

inside_set = [set_indicator(p) for p in sol.u] # use interpolation?
@show all(inside_set) # total survivability
@show findlast(cumprod(inside_set)) # finite time survival

sol.t[finite_surv]
plot(sol)
plot!(sol.t, inside_set)
hline!([-0.2, 0.2])
