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

@time for α in αs
    println(α)
    p = SwingPars(1.0, α, 8.0, 0.0)
    fp = [0.0, asin(p.P / p.K)]
    Tend = α < 0.1 ? 1000 : 500
    ode = ODEProblem(swing!, fp, (0.0, Tend), p)
    μ, μerr, converged = basin_stability_fixpoint(
        ode,
        fp,
        T,
        lb,
        ub;
        sample_alg = UniformSample(),
        distance = PeriodicEuclidean([Inf, 2π]),
        threshold = 1e-5,
        solver = Rosenbrock23(),
        parallel_alg = EnsembleThreads(),
    )
    push!(sobol_estimate, μ ± μerr)
end

begin
    plotd = scatter(αs,
        [se for se in sobol_estimate],
        #label = "Sobol sample",
        legend = false, #:topleft,
        #xscale=:log10
    )
    #title!("Basin stability estimation\n 95% confidence interval")
    ylabel!("basin stability S")
    xlabel!("dissipation \\alpha")
    vline!([0.1, 0.27, 0.3], c=:black, ls=:dot, label="")
end



## WIP draw phase space pictures
function basin_plot(sol, x0, y0, in_basin; kwargs...)
    n = length(sol(0)) ÷ 2
    p = scatter(
        sol,
        vars = [
            ((x, y) -> (mod2pi(x + π) - π, y), p + n, p)
            for p in findall(in_basin)
        ],
        shape = :rect,
        legend = false,
        c = :black,
        ms = 1,
        grid=false,
        markerstrokewidth = 0;
        kwargs...
    )
    if sum(in_basin) < n
        scatter!(
            p,
            sol,
            vars = [
                ((x, y) -> (mod2pi(x + π) - π, y), p + n, p)
                for p in findall(.!in_basin)
            ],
            shape = :rect,
            legend = false,
            c = :orange,
            ms = 1,
            grid=false,
            markerstrokewidth = 0;
            #bg_color = :red,
            kwargs...
        )
    end
    xlims!(extrema(x0)...)
    ylims!(extrema(y0)...)
    return p
end

rect(w, h, x, y) = Shape(x .+ [0, w, w, 0, 0], y .+ [0, 0, h, h, 0])

T = 200
lb = [-15.0, -π]
ub = - lb

begin
    p = SwingPars(1.0, 0.1, 8.0, 0.0)
    fp = [asin(p.P / p.K), 0.0]
    perts = perturbation_set_rect(fp, 1:2, T, lb, ub)
    θ0 = last.(perts)
    ω0 = first.(perts)
    ode = ODEProblem(swing!,[ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode, Rosenbrock23())
    final_freq = sol[end][1:length(ω0)]
    converged = abs.(final_freq) .< 0.1
end

plota = basin_plot(sol, θ0, ω0, converged, denseplot=true, alpha=1., ms=1., xlabel="phase \\theta", ylabel="frequency \\omega");
    plot!(rect(2.2, 4.5, 0.9, 8), fillcolor=:white);
    annotate!([(1., 10, Plots.text("\\alpha=$(p.α) \\nS=$(sobol_estimate[findfirst(αs .== p.α)])", 6, :black, :left)),]);

begin
    p = SwingPars(1.0, 0.27, 8.0, 0.0)
    fp = [asin(p.P / p.K), 0.0]
    perts = perturbation_set_rect(fp, 1:2, T, lb, ub)
    θ0 = last.(perts)
    ω0 = first.(perts)
    ode = ODEProblem(swing!, [ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode)
    converged = abs.(sol[end][1:length(ω0)]) .< 0.1
end

plotb = basin_plot(sol, θ0, ω0, converged, denseplot=true, alpha=1., ms=1., xlabel="phase \\theta", ylabel="frequency \\omega");
    plot!(rect(2.2, 4.5, 0.9, 8), fillcolor=:white);
    annotate!([(1., 10, Plots.text("\\alpha=$(p.α) \\nS=$(sobol_estimate[findfirst(αs .== p.α)])", 6, :black, :left)),]);

begin
    p = SwingPars(1.0, 0.28, 8.0, 0.0)
    fp = [asin(p.P / p.K), 0.0]
    perts = perturbation_set_rect(fp, 1:2, T, lb, ub)
    θ0 = last.(perts)
    ω0 = first.(perts)
    ode = ODEProblem(swing!, [ω0; θ0], (0.0, 100.0), p)
    sol = solve(ode)
    converged = abs.(sol[end][1:length(ω0)]) .< 0.1
end

plotc = basin_plot(sol, θ0, ω0, converged, denseplot=true, alpha=1., ms=1., xlabel="phase \\theta", ylabel="frequency \\omega");
    plot!(rect(2.2, 4.5, 0.9, 8), fillcolor=:white);
    annotate!([(1., 10, Plots.text("\\alpha=$(p.α) \\nS=$(sobol_estimate[findfirst(αs .== p.α)])", 6, :black, :left)),]);

plot(plota, plotb, plotc, plotd; size=(1200, 200), dpi=300, layout=@layout [a b c d{0.4w}])
savefig("$(@__DIR__)/Menck2012_fig1.png")

## network analysis

"""
Now, we want to reproduce Fig. 2 of

Menck,P & Kurths, J.
Topological Identification of Weak Points in Power Grids.
Nonlinear Dynamics of Electronic Systems, 144-147, 2012
"""

@inline Base.@propagate_inbounds function kuramoto_edge!(e,v_s,v_d,p,t)
    e[1] = 8. * sin(v_s[2] - v_d[2])
    nothing
end

@inline Base.@propagate_inbounds function kuramoto_vertex!(dv, v, e_s, e_d, p, t)
    dv[1] = p - 0.1v[1]
    for e in e_s
        dv[1] -= e[1]
    end
    for e in e_d
        dv[1] += e[1]
    end
    dv[2] = v[1]
    nothing
end

odevertex = ODEVertex(f! = kuramoto_vertex!, dim = 2, sym=[:ω, :ϕ])
staticedge = StaticEdge(f! = kuramoto_edge!, dim = 1)


T = 10 # 500
lb = [-100.0, -π]
ub = - lb

num_grids = 5 # 5783

df = DataFrame()

for _ in 1:num_grids
    g = erdos_renyi(64, 264) |> kruskal_mst |> SimpleGraph
        while ne(g) < 80
            cand = rand(vertices(g), 2)
            if !has_edge(g, cand...)
                add_edge!(g, cand...)
            end
        end
        @assert is_connected(g)




    N = nv(g)
    P = ones(N)
    P[1:2:N] .*= -1
    #sample(1:N, N÷2; replace=false)

    @assert iszero(sum(P))

    p = (P, nothing)

    nd = network_dynamics([odevertex for v in vertices(g)], [staticedge for e in edges(g)], g; parallel=true)
        guess = zeros(2nv(g))
        res = nlsolve((dx, x) -> nd(dx, x, p, 0.), guess)
        fp = res.zero

    @assert NLsolve.converged(res)

    # perts = perturbation_set_uniform(fp, 1:2, T, lb, ub)
    # ode = ODEProblem(nd, perts[:, 7], (0., 500.), p)
    # sol = solve(ode, AutoTsit5(Rosenbrock23()))
    # plot(sol, vars=idx_containing(nd, :ω), legend=false)
    # [evaluate(distance, fp, u) for u in sol.u] |> x-> plot(sol.t, x, yscale=:log10)


    # To follow the method in the paper, we ignore the phase distance, i.e. different
    # fixpoints are not distinguished given the frequency vanishes.
    dimension_weights = ones(2N)
    dimension_weights[2:2:2N] .= 0
    distance = WeightedEuclidean(dimension_weights)

    uniform_estimate = []
        @time for (node, node_idx) in enumerate(nd.f.graph_structure.v_idx)
            println(node)
            μ, μerr, converged = basin_stability_fixpoint(
                ode,
                fp,
                T,
                lb,
                ub;
                dimensions=node_idx,
                sample_alg = UniformSample(),
                distance = distance,
                threshold = 1e-2,
                #solver = RadauIIA5(),
                parallel_alg = EnsembleThreads(),
            )
            push!(uniform_estimate, μ±μerr)
        end

    _df = DataFrame(
        p = P,
        d = degree(g),
        nd = [degree(g, neighbors(g, v)) |> mean |> round for v in vertices(g)], # binning to integers
        bw = betweenness_centrality(g, normalize=false) .|> round ,# binning to integers
        bs = [ue for ue in uniform_estimate],
        bsm = [ue.val for ue in uniform_estimate],
    )
    append!(df, _df)
end

# Fig 2a
@df df histogram(:bsm, nbins=:auto, legend=false, xlabel="single-node basin stability S", ylabel="density")

# Fig 2b
gd = combine(groupby(df, :d), :bs => mean)
    scatter(gd.d, gd.bs_mean, legend=false, xlabel="nodal degree d", ylabel="expected basin stability E (S | d, d^N)")

# Fig 3a
gd = combine(groupby(df[df.d .== 1,:], :nd), :bs => mean) |> sort
    plot(gd.nd, gd.bs_mean, xlabel="average neighbour degree d^N", ylabel="expected basin stability E (S | d, d^N)", label="d=1")
    gd = combine(groupby(df[df.d .== 2,:], :nd), :bs => mean) |> sort
    plot!(gd.nd, gd.bs_mean, label="d=2")

# Fig 3b
gd = combine(groupby(df[df.d .== 3,:], :nd), :bs => mean) |> sort
    plot(gd.nd, gd.bs_mean, xlabel="average neighbour degree d^N", ylabel="expected basin stability E (S | d, d^N)", label="d=3")
    gd = combine(groupby(df[df.d .== 4,:], :nd), :bs => mean) |> sort
    plot!(gd.nd, gd.bs_mean, label="d=4")

# Fig 4a
gd = combine(groupby(df, :bw), :bs => mean) |> sort
    plot(gd.bw, gd.bs_mean, legend=false, xlims=(0, 400), xlabel="betweenness b", ylabel="expected basin stability E (S | b)")
    vline!([N-2, 2(N-3), 2(N-3)+1, 3(N-4)+2, 4(N-5)+3, 5(N-6)+4], c=:black, ls=:dot)
