using Pkg
Pkg.activate(".")

function return_trajectories(sol, i) #output_func
    (sol, false)
end

function all_evals(sol, i) #output_func
    a = eval_convergence_to_state(sol, fp, euclidean; verbose = false)
    b = eval_final_distance_to_state(sol, fp, euclidean; verbose = false)
    c = eval_trajectory_within_bounds(
        sol,
        [-2.0, -Inf],
        [2.0, Inf];
        verbose = false,
    )
    ([a; b; c], false)
end

# basin_stability_fixpoint(
#     f, u0, tspan, p,
#     fixpoint,
#     sample_size,
#     lb,
#     ub;
#     dimensions=:,
#     threshold=1E-4,
#     parallel_alg = nothing,
#     solver = nothing,
#     sample_alg = nothing,
#     verbose = false,
# ) = basin_stability_fixpoint(
#     ODEProblem(f, u0, tspan, p),
#     fixpoint,
#     sample_size,
#     lb,
#     ub;
#     dimensions=dimensions,
#     threshold=threshold,
#     parallel_alg = nothing,
#     solver = nothing,
#     sample_alg = nothing,
#     verbose = verbose,
# )



# ODEProblem
function basin_stability_fixpoint(
    ode_prob::ODEProblem,
    fixpoint,
    sample_size,
    lb,
    ub;
    dimensions=:,
    distance=Euclidean(),
    threshold=1E-4,
    parallel_alg = nothing,
    solver = nothing,
    sample_alg = nothing,
    verbose = false,
)

    if isnothing(sample_alg)
        ics = perturbation_set_sobol(fixpoint, dimensions, sample_size, lb, ub; verbose=verbose)
    else
        ics = perturbation_set(fixpoint, dimensions, sample_size, lb, ub, sample_alg, verbose)
    end

    #TODO: pass solve args through
    close, converged = basin_stability_MCsample(
        ode_prob,
        fixpoint,
        sample_size,
        ics;
        distance=distance,
        threshold=threshold,
        parallel_alg = parallel_alg,
        solver = solver,
        verbose = verbose,
        )

    if verbose
        println(count(close), " initial conditions arrived close to the fixpoint (threshold $threshold) ", count(converged), " indicate convergence.")
    end

    return sample_statistics(close)
end

# ODEProblem, custom ICset
function basin_stability_MCsample(
    ode_prob::ODEProblem,
    fixpoint,
    sample_size,
    ics::Union{ICset, Array};
    distance=Euclidean(),
    threshold=1E-4,
    parallel_alg = nothing,
    solver = nothing,
    verbose = false,
)
    #TODO: pass solve args through

    if verbose
        println("Parallel ensemble simulation")
    end

    option_s = isnothing(solver) ? AutoTsit5(Rosenbrock23()) : solver
    option_p = isnothing(parallel_alg) ?
        (nprocs() > 1 ? EnsembleThreads() : EnsembleSerial()) :
        parallel_alg

    #(prob,i,repeat)->(prob)
    prob_func(prob, i, repeat) = remake(prob, u0 = ics[:, i])

    #(sol,i) -> (sol,false)
    function all_evals(sol, i) #output_func
        co = eval_convergence_to_state(
            sol,
            fixpoint,
            distance;
            verbose = false,
        )
        di = eval_final_distance_to_state(
            sol,
            fixpoint,
            distance; # per dimension, use state_filter?
            threshold = threshold,
            verbose = false,
        )
        ([co; di], false)
    end

    eprob = EnsembleProblem(
        ode_prob;
        output_func = all_evals, #(sol,i) -> (sol,false),
        prob_func = prob_func, #(prob,i,repeat)->(prob),
        #reduction = (u,data,I)->(append!(u,data),false),
        u_init = [],
    )

    if verbose
        @time esol = solve(
            eprob,
            option_s,
            option_p,
            #saveat = 0.1,
            trajectories = sample_size,
            callback = TerminateSteadyState(1E-8, 1E-6), #
        )
    else
        esol = solve(
            eprob,
            option_s,
            option_p,
            #saveat = 0.1,
            trajectories = sample_size,
            callback = TerminateSteadyState(1E-8, 1E-6), # 1E-8, 1E-6
        )
    end

    converged = first.(esol.u)
    close = last.(esol.u)

    return close, converged
end

# DynamicalSystem
function basin_stability_fixpoint(
    ds::DynamicalSystem,
    fixpoint,
    sample_size,
    lb,
    ub;
    dimensions=:,
    distance=Euclidean(),
    threshold=1E-4,
    Tend=100,
    solver = nothing,
    sample_alg = nothing,
    verbose = false,
)
    #TODO: pass solve args through

    if verbose
        println("Parallel integrator simulation")
    end

    if isnothing(sample_alg)
        ics = perturbation_set_sobol(fixpoint, dimensions, sample_size, lb, ub; verbose=verbose)
    else
        ics = perturbation_set(fixpoint, dimensions, sample_size, lb, ub, sample_alg, verbose)
    end

    option_s = isnothing(solver) ? Tsit5() : solver

    # the TerminateSteadyState callback does not help here since all ics are evaluated at the same time points
    kwargs = (alg = option_s,)# callback=AutoAbstol(), abstol=1e-14, reltol=1e-14, maxiters=1e9)
    states = [SVector{size(fixpoint)...}(ics[:, k]) for k in 1:sample_size]
    pint = parallel_integrator(ds, states; kwargs...)

    # TODO: more elaborated stepping to evaluate other observables, i.e. convergence...

    if verbose
        @time while pint.t < Tend
            step!(pint)
        end
    else
        while pint.t < Tend
            step!(pint)
        end
    end

    close = eval_final_distance_to_state(pint, fixpoint, distance; threshold=threshold)

    if verbose
        println(count(close), " initial conditions arrived close to the fixpoint (threshold $threshold).")
    end

    return sample_statistics(close)
end
