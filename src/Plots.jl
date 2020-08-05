
function basin_plot_trajectories(
    ode_prob::ODEProblem,
    fixpoint,
    sample_size,
    lb,
    ub;
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
        sample_size,
        ics;
        distance = distance,
        threshold = threshold,
        parallel_alg = parallel_alg,
        solver = solver,
        verbose = verbose,
        )


    fig = plot()
    v1 = first(dimensions)
    v2 = last(dimensions)

    in_basin = first.(esol)

    # draw out of basin
    for (in_basin, sol) in esol
        scatter!(
            fig,
            sol,
            vars = angular_axis(periodic_dim, v1, v2),
            shape = :rect,
            legend = false,
            c = in_basin ? :orange : :black,
            ms = 1,
            grid = false,
            markerstrokewidth = 0;
            plot_kwargs...
        )
    end

    xlims!(lb[1], ub[1])
    ylims!(lb[2], ub[2])

    return fig
end

function angular_axis(periodic_dim, v1, v2)
    if periodic_dim == 1
        vars = ((x, y) -> (mod2pi(x + π) - π, y), v1, v2)
    elseif periodic_dim == 2
        vars = ((x, y) -> (mod2pi(x + π) - π, y), v1, v2)
    elseif isa(periodic_dim, Colon)
        vars = ((x, y) -> (mod2pi(x + π) - π, mod2pi(y + π) - π), v1, v2)
    else
        vars = (v1, v2)
    end
end
