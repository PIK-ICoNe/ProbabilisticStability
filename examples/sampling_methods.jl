using Pkg
Pkg.activate(@__DIR__)
using ProbabilisticStability
using OrdinaryDiffEq
using QuasiMonteCarlo
using Random
Random.seed!(42);
using BenchmarkTools
using UnicodePlots

## example system: damped-driven oscillator

struct SwingPars
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

## compare MonteCarlo and QuasiMonteCarlo sampling

T = 500

p = SwingPars(1.0, 0.1, 8.0, 0.0)
fp = [0.0, asin(p.P / p.K)] # we know the stable fixpoint

# define sampling region
lb = [-100.0, -π]
ub = -lb

tspan = (0., 200.)
ode = ODEProblem(swing!, fp .+ randn(2),  tspan, p)

sol = solve(ode, Rodas4())

# plot the frequency
pl = lineplot(sol.t, [first(sol(t)) for t in sol.t]);
display(pl)

@btime uniform_ics = perturbation_set(fp, :, T, lb, ub, UniformSample(), false)
@btime sobol_ics = perturbation_set(fp, :, T, lb, ub, SobolSample(), false)
@btime lhc_ics = perturbation_set(fp, :, T, lb, ub, LatinHypercubeSample(), false)
