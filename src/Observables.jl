function get_convergence_to_state(sol::AbstractODESolution, state, distance; tail_frac=0.8)
    L = length(sol.t)
    idx = max(1, round(Int, tail_frac*L)):L
    x = sol.t[idx]
    y = [evaluate(distance, state, p) for p in sol.u[idx]]
    X = zeros(length(x),2)
    X[:,1] = x
    X[:,2] .= 1.0
    slope, intercept = X \ log.(y)
    return slope
end

function eval_convergence_to_state(sol::AbstractODESolution, state, distance; tail_frac=0.8, verbose=false)
    slope = get_convergence_to_state(sol, state, distance; tail_frac=tail_frac)
    if verbose
        println("The estimated slope is $slope.")
    end
    return slope < 0
end

function get_final_distance_to_state(sol::AbstractODESolution, state, distance; threshold=1E-3, verbose=true)
    return evaluate(distance, state, sol[end])
end

function eval_final_distance_to_state(sol::AbstractODESolution, state, distance; threshold=1E-3, verbose=true)
    d = evaluate(distance, state, sol[end])
    if verbose
        println("The final state distance is $d.")
    end
    return d < threshold
end

function eval_final_distance_to_state(d::Real; threshold=1E-3, verbose=true)
    if verbose
        println("The final state distance is $d.")
    end
    return d < threshold
end

"""
e.g. distance -> PeriodicEuclidean([Inf,2π]) or Euclidean()
"""
function eval_final_distance_to_state(pint, state, distance; threshold=1E-3, verbose=true)
    d = colwise(distance, state, pint.u)
    if verbose
        println("The final state distance is in the range: ", extrema(d))
    end
    return d .< threshold
end

# function eval_final_distances_to_state(sol, state, distance; state_filter=nothing, threshold=1E-3, verbose=true)
#     d = distance.(state, sol[end])
#     if isnothing(state_filter)
#         dmax = maximum(d)
#     else
#         dmax = maximum(d[state_filter])
#     end
#     if verbose
#         println("The max final state distance across all dimensions is $dmax.")
#     end
#     return dmax < threshold
# end

function get_mean_distance_to_state(sol::AbstractODESolution, state, distance; tail_frac=0.8)
    L = length(sol.t)
    idx = max(1, round(Int, tail_frac*L)):L
    return mean([distance(state, p) for p in sol.u[idx]])
end

function eval_mean_distance_to_state(sol::AbstractODESolution, state, distance; threshold=1E-3, tail_frac=0.8, verbose=true)
    d = get_mean_distance_to_state(sol, state, distance; tail_frac=tail_frac)
    if verbose
        println("The mean state distance is $d.")
    end
    return d < threshold
end

eval_mean_distance_to_state(d; threshold=1E-3) = d < threshold

function get_max_distances_to_state(sol::AbstractODESolution, state, distance; tail_frac=0.8)
    L = length(sol.t)
    #idx = max(1, round(Int, tail_frac*L)):L
    return maximum(distance.(fp, sol), dims=2)
end

function eval_max_distances_to_state(sol::AbstractODESolution, state, lb, ub, distance; tail_frac=0.8, verbose=true)
    d = get_max_distances_to_state(sol, state, distance; tail_frac=tail_frac)
    if verbose
        println("The max state distances are $d.")
    end
    return lb .< d .< ub
end

eval_max_distances_to_state(d, lb, ub) =  lb .< d .< ub

function eval_max_distance_to_state(sol::AbstractODESolution, state, lb, ub, distance; tail_frac=0.8, verbose=true)
    d = get_max_distances_to_state(sol, state, distance; tail_frac=tail_frac)
    if verbose
        println("The max state distance is $d.")
    end
    return all(lb .< d .< ub)
end

eval_max_distance_to_state(d, lb, ub) =  all(lb .< d .< ub)

function get_trajectory_within_bounds(sol::AbstractODESolution, lb, ub; tail_frac=0, verbose=true)
    L = length(sol.t)
    idx = max(1, round(Int, tail_frac*L)):L
    return [all(lb .< p .< ub) for p in sol.u[idx]]
end

function eval_trajectory_within_bounds(sol::AbstractODESolution, lb, ub; tail_frac=0, verbose=true)
    # use Inf as bounds for dimension that should be beglected
    ep = get_trajectory_within_bounds(sol, lb, ub; tail_frac=tail_frac, verbose=verbose)
    #findlast(.! ep)
    return all(ep)
end

eval_trajectory_within_bounds(twb::Array{Bool,1}) = all(twb)

function all_evals(sol, i) # output_func
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
