using Pkg
Pkg.activate(@__DIR__)
using Revise

using ProbabilisticStability

using OrdinaryDiffEq
using DiffEqCallbacks
using DynamicalSystems
using MCBB
using Distributions, Clustering
using Distances
using QuasiMonteCarlo # to compare different samplings
using Measurements # nice uncertainty handling
using NetworkDynamics
using LightGraphs

using Random
Random.seed!(42);

using BenchmarkTools
using UnicodePlots

"""
Here, we want to reproduce step-by-step Fig. 1 from the publication:

Menck,P & Kurths, J.
Topological Identification of Weak Points in Power Grids.
Nonlinear Dynamics of Electronic Systems, 144-147, 2012
"""

struct SwingPars #<: DEParameters
    P
    α
    K
    θgrid
end

function swing!(du, u, p, t)
    n = length(u) ÷ 2
    ωs = 1:n
    θs = n+1:2n
    @. du[ωs] = p.P - p.α * u[ωs] - p.K .* sin.(u[θs] - p.θgrid)
    @. du[θs] = u[ωs]
end



## compare MonteCarlo and QuasiMonteCarlo sampling, parallel_integrator and EnsembleProblem

T = 500

p = SwingPars(1.0, 0.1, 8.0, 0.0)
fp = [0.0, asin(p.P / p.K)] # we know the stable fixpoint

# define sampling region
lb = [-100.0, -π]
ub = -lb

ode = ODEProblem(swing!, fp,  (0., 200.), p)

# for small systems, it could be more performant to stack them up?
ds = ContinuousDynamicalSystem(ode)
@btime basin_stability_fixpoint(ds, fp, T, lb, ub; verbose=false, distance=PeriodicEuclidean([Inf,2π]), threshold=1.)
# 16.589 s (91124105 allocations: 3.50 GiB)

# EnsembleProblem version
@btime basin_stability_fixpoint(ode, fp, T, lb, ub; verbose=false, distance=PeriodicEuclidean([Inf,2π]), threshold=1.)
# 5.752 s (48351905 allocations: 2.00 GiB)


mle_estimate = []
uniform_estimate = []
sobol_estimate = []
Ts = 100:50:1000
for T in Ts
    println(T)
    μ, μerr, converged = basin_stability_fixpoint(ode, fp, T, lb, ub; sample_alg=UniformSample(), distance=PeriodicEuclidean([Inf,2π]), threshold=1., verbose=false)
    push!(uniform_estimate, μ ± μerr)
    push!(mle_estimate, converged / T)
    μ, μerr, converged = basin_stability_fixpoint(ode, fp, T, lb, ub; verbose=false, distance=PeriodicEuclidean([Inf,2π]), threshold=1.)
    push!(sobol_estimate, μ ± μerr)
end



begin
    scatter(Ts,
        [se for se in sobol_estimate],
        label = "Sobol sequence",
        legend = :best,
        #xscale=:log10
    )
    scatter!(Ts, mle_estimate; label = "MLE", marker=:cross)
    scatter!(Ts, [ue for ue in uniform_estimate], label = "uniform distribution")
    hline!([0.26], label = "Menck et al.")
    title!("Basin stability estimation, 95% confidence interval")
    ylabel!("basin stability μ")
    xlabel!("sample size T")
end

begin
    scatter(Ts,
        [se - last(sobol_estimate) for se in sobol_estimate] ./ last(sobol_estimate),
        label = "Sobol sequence",
        legend = :best,
        #xscale=:log10
    )
    scatter!(Ts, [ue - last(uniform_estimate) for ue in uniform_estimate] ./ last(uniform_estimate), label = "uniform distribution")
    title!("Basin stability estimation, 95% confidence interval")
    ylabel!("relative deviation from best estimate %")
    xlabel!("sample size T")
end

## WIP draw phase space pictures

T = 5000
fp = [asin(p.P / p.K), 0.0]
lb = [-π, -15.0]

# I'm not happy with the grid, need to find a better solution
perts = single_node_perturbation_grid(fp, 1:2, T, 2, lb, -lb)

# a better way would be an interpolation/binning to a regular grid
begin
    p = SwingPars(1.0, 0.1, 8.0, 0.0)
    θ0 = fp[2] .+ Iterators.flatten([first(p) for p in perts])
    ω0 = fp[1] .+ Iterators.flatten([last(p) for p in perts])
    ode = ODEProblem(swing!, [ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode)
    converged = abs.(sol[end][1:length(ω0)]) .< 0.1
end

color_code = map(x -> x ? "green" : "red", converged)
scatter(θ0, ω0, mc = color_code)

# TODO: check this out
#contourf(ics[2, :], ics[1, :], c, level=2)
#color_palette = cgradient(["red", "green"])

begin
    p = SwingPars(1.0, 0.27, 8.0, 0.0)
    θ0 = fp[2] .+ Iterators.flatten([first(p) for p in perts])
    ω0 = fp[1] .+ Iterators.flatten([last(p) for p in perts])
    ode = ODEProblem(swing!, [ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode)
    converged = abs.(sol[end][1:length(ω0)]) .< 0.1
end

color_code = map(x -> x ? "green" : "red", converged)
scatter(θ0, ω0, mc = color_code)

begin
    p = SwingPars(1.0, 0.28, 8.0, 0.0)
    θ0 = fp[2] .+ Iterators.flatten([first(p) for p in perts])
    ω0 = fp[1] .+ Iterators.flatten([last(p) for p in perts])
    ode = ODEProblem(swing!, [ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode)
    converged = abs.(sol[end][1:length(ω0)]) .< 0.1
end

color_code = map(x -> x ? "green" : "red", converged)
scatter(θ0, ω0, mc = color_code)




## Idea collection

fp = [0.0, asin(p.P / p.K)]

T = 2000

ode = ODEProblem(swing!, fp, (0.0, 200.0), p)

sol = remake(ode; u0 = fp.+[.1, 0.1]) |> solve

plot(sol, vars = (2, 1), plotdensity=100)

lb = [-0.2, -Inf]
ub = [0.2, Inf]

eval_convergence_to_state(sol, fp, euclidean; verbose=true)
eval_final_distance_to_state(sol, fp, euclidean; verbose=true)
eval_mean_distance_to_state(sol, fp, euclidean; verbose=true)
eval_max_distance_to_state(sol, fp, lb, ub, euclidean; verbose=true)
eval_max_distances_to_state(sol, fp, lb, ub, euclidean; verbose=true)
eval_trajectory_within_bounds(sol, lb, ub; verbose=true)

d = get_max_distances_to_state(sol, fp, euclidean)



plot(sol, vars=1)
plot!(sol.t, first.(ep))
hline!([ub[1],])

##

ics =
    fp .+
    single_node_perturbation_sobol(fp, 1:2, T, 2, [-100.0, -π], [100.0, π])

pars = rand(Uniform(0.0, 0.4), T)

ic_gens = (i_run) -> ics[:, i_run]
par_gens = (i_run) -> pars[i_run]

#TODO write observables for stability measures

function eval_ode_run_kura(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim÷2)
    eval_funcs = [mean, std]
    matrix_eval_funcs = []
    global_eval_funcs = []
    eval_ode_run(
        sol,
        i,
        state_filter,
        eval_funcs,
        matrix_eval_funcs,
        global_eval_funcs,
        cyclic_setback = false,
    )
end

mcp = DEMCBBProblem(ode, ic_gens, T, p, (:α, par_gens), eval_ode_run_kura, 0.8)

@time mcsol = solve(mcp)

#  * first: all per dimension measures in the same order as in the eval_funcs array (default: 1: mean, 2: SD, 3: KL-Div) * then: all matrix measures * then: all global measures * optional: for routines that also incorporate the parameters, they are last in order.
# For the example above [1.,0.75,0.,1.] thus means: weight 1. on mean, 0.75 on SD, 0. on KL-Div and 1. on the parameter
D = distance_matrix(mcsol, mcp, [1.0, 0.0, 0.0, 0.0], histograms = false)

db_eps = median(KNN_dist_relative(D))
db_minpts = round(Int, log(T))
db_res = dbscan(D, db_eps, db_minpts)

# TODO write alternative to cluster_membership
cluster_members = cluster_membership(mcp, db_res, 0.05, 0.005)

plot(cluster_members, legend = true)


# need sth like 'basin_stability_on_parameter_sliding_window'
