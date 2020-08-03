
function return_trajectories(sol, i) # output_func
    (sol, false)
end

# ODEProblem, custom ICset
function mc_sample_from_IC(
    ode_prob::ODEProblem,
    eval_func::Function,
    sample_size::Int,
    ics::Union{ICset,Array};
    distance = Euclidean(),
    threshold = 1E-4,
    parallel_alg = nothing,
    solver = nothing,
    verbose = false,
)
    # TODO: pass solve args through

    if verbose
        println("Parallel ensemble simulation")
    end

    option_s = isnothing(solver) ? AutoTsit5(Rosenbrock23()) : solver
    option_p = isnothing(parallel_alg) ?
        (nprocs() > 1 ? EnsembleThreads() : EnsembleSerial()) :
        parallel_alg

    # (prob,i,repeat)->(prob)
    prob_func(prob, i, repeat) = remake(prob, u0 = ics[:, i])

    eprob = EnsembleProblem(
        ode_prob;
        output_func = eval_func, # (sol,i) -> (sol,false),
        prob_func = prob_func, # (prob,i,repeat)->(prob),
        # reduction = (u,data,I)->(append!(u,data),false),
        u_init = [],
    )

    if verbose
        @time esol = solve(
            eprob,
            option_s,
            option_p,
            # saveat = 0.1,
            trajectories = sample_size,
            callback = TerminateSteadyState(1E-8, 1E-6), #
        )
    else
        esol = solve(
            eprob,
            option_s,
            option_p,
            # saveat = 0.1,
            trajectories = sample_size,
            callback = TerminateSteadyState(1E-8, 1E-6), # 1E-8, 1E-6
        )
    end

    return esol.u
end



# ODEProblem
function basin_stability_fixpoint(
    ode_prob::ODEProblem,
    fixpoint,
    sample_size,
    lb,
    ub;
    dimensions = :,
    distance = Euclidean(),
    threshold = 1E-4,
    parallel_alg = nothing,
    solver = nothing,
    sample_alg = nothing,
    verbose = false,
)

    if isnothing(sample_alg)
        ics = perturbation_set_sobol(fixpoint, dimensions, sample_size, lb, ub; verbose = verbose)
    else
        ics = perturbation_set(fixpoint, dimensions, sample_size, lb, ub, sample_alg, verbose)
    end

    # (sol,i) -> (sol,false)
    function eval_func(sol, i) # output_func
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

    # TODO: pass solve args through
    esol = mc_sample_from_IC(
        ode_prob,
        eval_func,
        sample_size,
        ics;
        distance = distance,
        threshold = threshold,
        parallel_alg = parallel_alg,
        solver = solver,
        verbose = verbose,
        )

    converged = first.(esol)
    close = last.(esol)

    if verbose
        println(count(close), " initial conditions arrived close to the fixpoint (threshold $threshold) ", count(converged), " indicate convergence.")
    end

    return sample_statistics(close)
end

# DynamicalSystem
function basin_stability_fixpoint(
    ds::DynamicalSystem,
    fixpoint,
    sample_size,
    lb,
    ub;
    dimensions = :,
    distance = Euclidean(),
    threshold = 1E-4,
    Tend = 100,
    solver = nothing,
    sample_alg = nothing,
    verbose = false,
)
    # TODO: pass solve args through

    if verbose
        println("Parallel integrator simulation")
    end

    if isnothing(sample_alg)
        ics = perturbation_set_sobol(fixpoint, dimensions, sample_size, lb, ub; verbose = verbose)
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

    close = eval_final_distance_to_state(pint, fixpoint, distance; threshold = threshold)

    if verbose
        println(count(close), " initial conditions arrived close to the fixpoint (threshold $threshold).")
    end

    return sample_statistics(close)
end


## ODEProblem
function survivability(
    ode_prob::ODEProblem,
    indicator_func::Function, # an indicator function with signature p -> bool for a point p
    sample_size::Int,
    lb,
    ub;
    dimensions = :,
    parallel_alg = nothing,
    solver = nothing,
    sample_alg = nothing,
    verbose = false,
)

    if isnothing(sample_alg)
        # sample around origin since we don't need a fixpoint here
        ics = perturbation_set_sobol(zero(ode.u0), dimensions, sample_size, lb, ub; verbose = verbose)
    else
        ics = perturbation_set(zero(ode.u0), dimensions, sample_size, lb, ub, sample_alg, verbose)
    end

    function eval_func(sol, i) # output_func
        inside_set = [indicator_func(p) for p in sol.u] # use interpolation?
        surv = all(inside_set) # total survival
        finite_surv = findlast(cumprod(inside_set)) # finite time survival
        ([surv; isnothing(finite_surv) ? nothing : sol.t[finite_surv]], false)
    end

    # TODO: pass solve args through
    esol = mc_sample_from_IC(
        ode_prob,
        eval_func,
        sample_size,
        ics;
        distance = distance,
        threshold = threshold,
        parallel_alg = parallel_alg,
        solver = solver,
        verbose = verbose,
        )

    surv = first.(esol)
    finite_surv = last.(esol)

    if verbose
        println(count(surv), " initial conditions survived.")
    end

    return sample_statistics(surv)
end

