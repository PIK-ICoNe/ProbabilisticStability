using DiffEqUncertainty, Quadrature, OrdinaryDiffEq, Distributions, BenchmarkTools

# Define a system [Menck, 2012]
function swing!(du, u, p, t)
    du[1] = u[2] # phase
    du[2] = -p[1] * u[2] + p[2] - p[3] * sin(u[1]) # frequency
end

tspan = (0.0, 1000.0)
p = [0.1, 1, 8] # α, P, K
u_fix = [asin(p[2] / p[3]); 0.]
prob = ODEProblem(swing!, u_fix, tspan, p)

# Distribution of initial conditions
u0_dist = [Uniform(u_fix[1] - pi, u_fix[1] + pi), Uniform(-100, 100)]
u0_dist_small = [Uniform(u_fix[1] - pi, u_fix[1] + pi), Uniform(-1, 1)]

# Define an Observable for basin stability
# An initial condition belongs to the basin if its frequency converges to 0
# Here we assume a trajectory is converged if its end point deviates from 0 by less than 0.1

tail_frac = 0.8

##
converged(sol) = [
    mean(sol[2,tail_frac * end:end]) < 0.1 , 
    mean(sol[2,tail_frac * end:end]) > 5 
    ]
##
survivability(sol) = maximum(abs, sol[2,:]) < 1.
##
function naive_return_time(sol) 
    idx = findfirst( abs.(sol[2,:]) .< 0.1 )
    return isnothing(idx) ? last(sol.t) :  sol.t[idx]
end
##

mc = expectation(converged, prob, u0_dist, p, MonteCarlo(), Rodas4(); trajectories=1_000)
@assert sum(mc) ≈ 1

sv = expectation(survivability, prob, u0_dist_small, p, MonteCarlo(), Rodas4(); trajectories=1_000)

ϵ = 1. # smaller values do not work
rt = expectation(naive_return_time, prob, u0_dist, p, Koopman(), Rodas4(); iabstol=ϵ, ireltol=ϵ)


## MonteCarlo experiment to compute basin stability
@btime expectation(converged, prob, u0_dist, p, MonteCarlo(), Tsit5(); trajectories=1_000)
# 3.161 s (35140338 allocations: 3.32 GiB)
# 0.249

function bump(sol)
    z = abs(sol[2,end])
    if z < .1
        return 1
    elseif z < .2
        return exp(-1 / (1 - (z - .1)^2))
    else
        return 0
    end
end

@time quad  = expectation(bump, prob, u0_dist, p, Koopman(), Tsit5(); quadalg=CubaDivonne(), iabstol=1e-3)

