"""
# Arguments
- `lb`: list of lower sampling bounds
- `ub`: list of upper sampling bounds
"""
function perturbation_set(
    state,
    idxs,
    sample_size,
    lb,
    ub,
    alg::QuasiMonteCarlo.SamplingAlgorithm,
    verbose,
)
    @assert length(lb) == length(ub) == length(idxs)
    # output format (d, n)
    perturbations = QuasiMonteCarlo.sample(sample_size, lb, ub, alg)
    if verbose
        println("Sampling initial conditions based on $alg.")
    end
    return ICset(state, perturbations, idxs)
end

function perturbation_set(
    state,
    idxs::Colon,
    sample_size,
    lb,
    ub,
    alg::QuasiMonteCarlo.SamplingAlgorithm,
    verbose,
)
    @assert length(lb) == length(ub) == length(state)
    # output format (d, n)
    perturbations = QuasiMonteCarlo.sample(sample_size, lb, ub, alg)
    if verbose
        println("Sampling initial conditions based on $alg.")
    end
    return state .+ perturbations
end

struct ICset{T}
    state::Array{T}
    perturbations::Array{T}
    idxs
    outdim
end

ICset(state, perturbations, idxs) = ICset(
    state,
    perturbations,
    idxs,
    (length(state), last(size(perturbations))),
)

Base.size(ics::ICset) = ics.outdim
Base.length(ics::ICset) = prod(size(ics))
function Base.getindex(ics::ICset, i, j)
    out = zeros(ics.outdim)
    out[ics.idxs, :] .+= ics.perturbations
    out .+= ics.state
    getindex(out, i, j)
end


perturbation_set_uniform(
    state,
    idxs,
    sample_size,
    lb,
    ub;
    verbose = false,
) = perturbation_set(
    state,
    idxs,
    sample_size,
    lb,
    ub,
    UniformSample(),
    verbose,
)

perturbation_set_sobol(
    state,
    idxs,
    sample_size,
    lb,
    ub;
    verbose = false,
) = perturbation_set(
    state,
    idxs,
    sample_size,
    lb,
    ub,
    SobolSample(),
    verbose,
)

"""
# Arguments
- `lb`: list of lower sampling bounds
- `ub`: list of upper sampling bounds
"""
function perturbation_set_grid(
    state,
    idxs,
    sample_size,
    lb,
    ub;
    verbose = false,
)
    @assert length(lb) == length(ub) == nodal_dim

    if lb isa Number
        dx = (ub - lb) / sample_size
        perturbations = vec(lb:dx:ub)
    else
        d = length(lb)
        # each dimension should have ⌊n^(1/d)⌋ points
        num_points = floor(Int, exp10(log10(sample_size) / d))
        dx = (ub .- lb) ./ num_points
        x = [lb[j]:dx[j]:ub[j] for j = 1:d]
        perturbations = Iterators.product(x...)
    end

    if verbose
        println("Regular grid of initial conditions.")
    end

    return perturbations
end
