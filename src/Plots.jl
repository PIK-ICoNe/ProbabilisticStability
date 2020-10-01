function angular_axis(periodic_dim, v1, v2; verbose = false)
    if periodic_dim == 1
        if verbose
            println("Angular x-axis. Applying mod2pi.")
        end
        vars = ((x, y) -> (mod2pi(x + π) - π, y), v1, v2)
    elseif periodic_dim == 2
        if verbose
            println("Angular y-axis. Applying mod2pi.")
        end
        vars = ((x, y) -> (mod2pi(x + π) - π, y), v1, v2)
    elseif isa(periodic_dim, Colon)
        if verbose
            println("Angular x/y-axis. Applying mod2pi.")
        end
        vars = ((x, y) -> (mod2pi(x + π) - π, mod2pi(y + π) - π), v1, v2)
    elseif isnothing(periodic_dim)
        if verbose
            println("Regular axis.")
        end
        vars = (v1, v2)
    end
end


function basin_plot_trajectories(
    ode_prob::ODEProblem,
    fixpoint,
    sample_size,
    lb,
    ub;
    plotfunc = scatter!,
    periodic_dim = nothing,
    dimensions = :,
    distance = Euclidean(),
    threshold = 1E-4,
    parallel_alg = nothing,
    solver = nothing,
    verbose = false,
    plot_kwargs...
    )

    @assert length(dimensions) == 2

    ics = perturbation_set_rect(fixpoint, dimensions, sample_size, lb, ub)

    # (sol,i) -> (sol,false)
    function eval_func(sol, i) # output_func
        di = eval_final_distance_to_state(
            sol,
            fixpoint,
            distance; # per dimension, use state_filter?
            threshold = threshold,
            verbose = false,
        )
        ([di, sol], false)
    end

    # TODO: pass solve args through
    esol = mc_sample_from_IC(
        ode_prob,
        eval_func,
        last(ics.outdim),
        ics;
        distance = distance,
        threshold = threshold,
        parallel_alg = parallel_alg,
        solver = solver,
        verbose = verbose,
    )


    fpx = fixpoint[first(dimensions)]
    fpy = fixpoint[last(dimensions)]

    fig = plot()
    v1 = first(dimensions)
    v2 = last(dimensions)

    # draw out of basin
    for (in_basin, sol) in esol
        plotfunc(
            fig,
            sol,
            vars = angular_axis(periodic_dim, v1, v2; verbose = false),
            shape = :rect,
            legend = false,
            c = in_basin ? :orange : :blue,
            ms = 1,
            grid = false,
            markerstrokewidth = 0;
            plot_kwargs...
        )
    end

    xlims!(fpx + lb[1], fpx + ub[1])
    ylims!(fpy + lb[2],  fpy + ub[2])

    return fig
end

function basin_plot(
    df::DataFrame,
    fixpoint,
    lb,
    ub;
    plotfunc = scatter!,
    periodic_dim = nothing,
    dimensions = :,
    verbose = false,
    plot_kwargs...
    )

    fpx = fixpoint[first(dimensions)]
    fpy = fixpoint[last(dimensions)]

    x = fpx .+ first.(df.perturbation)
    y = fpy .+ last.(df.perturbation)
    c = [ v ? :orange : :blue for v in df.within_threshold]

    if periodic_dim == 1
        x .= mod2pi(x + π) - π
    elseif periodic_dim == 2
        y .= mod2pi(y + π) - π
    elseif isa(periodic_dim, Colon)
        x .= mod2pi(x + π) - π
        y .= mod2pi(y + π) - π
    end

    fig = plot()
    plotfunc(
        fig,
        x, y, c=c, 
        shape = :rect,
        legend = false,
        ms = 1,
        grid = false,
        markerstrokewidth = 0;
        plot_kwargs...
        )

    xlims!(fpx + lb[1], fpx + ub[1])
    ylims!(fpy + lb[2],  fpy + ub[2])

    return fig
end
