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

# function perturbation_set(
#     state,
#     idxs::Colon,
#     sample_size,
#     lb,
#     ub,
#     alg::QuasiMonteCarlo.SamplingAlgorithm,
#     verbose,
# )
#     @assert length(lb) == length(ub) == length(state)
#     # output format (d, n)
#     perturbations = QuasiMonteCarlo.sample(sample_size, lb, ub, alg)
#     if verbose
#         println("Sampling initial conditions based on $alg.")
#     end
#     return state .+ perturbations
# end

struct ICset #{T, N}
    state#::Array{T, 1}
    perturbations#::Array{T, N}
    idxs#::Union{UnitRange, AbstractArray}
    outdim#::Tuple
end

function ICset(state, perturbations, idxs)  
    @assert length(idxs) == first(size(perturbations))
    #sstate = SVector{length(state)}(state)
    #sperturbations = SMatrix{size(perturbations)...}(perturbations)
    return ICset(
        state,
        perturbations,
        idxs,
        (length(state), last(size(perturbations)))
    )
end

##
Base.size(ics::ICset) = ics.outdim
Base.length(ics::ICset) = prod(size(ics))

function Base.getindex(ics::ICset, i::Int)
    out = zeros(eltype(ics.state), first(ics.outdim)) 
    out .+= ics.state
    patch = @view out[ics.idxs, :]
    patch .+= ics.perturbations[:, i]
    return out
end

Base.getindex(ics::ICset, i::Colon, j::Int) = Base.getindex(ics, j)

function Base.getindex(ics::ICset, i::Union{Int, UnitRange}, j::Union{Int, UnitRange})
    out = zeros(eltype(ics.state), first(ics.outdim), length(j)) 
    out .+= ics.state
    patch = @view out[ics.idxs, :]
    patch .+= ics.perturbations[:, j]
    return getindex(out, i, :) 
end

Base.getindex(ics::ICset, i::Colon, j::UnitRange) = Base.getindex(ics, 1:first(ics.outdim), j)
Base.getindex(ics::ICset, j::UnitRange) = Base.getindex(ics, :, j)

function Base.getindex(ics::ICset, i::Int, j::Colon)
    out = zeros(eltype(ics.state), ics.outdim...) 
    out .+= ics.state
    patch = @view out[ics.idxs, :]
    patch .+= ics.perturbations[:, j]
    return getindex(out, i, :)
end
##

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
    @assert length(lb) == length(ub)

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


"""
this functions creates a 2D rectangle, ideal for phase space plots

# Arguments
- `lb`: list of lower sampling bounds
- `ub`: list of upper sampling bounds
"""
function perturbation_set_rect(
    state,
    idxs,
    sample_size,
    lb,
    ub;
    verbose = false,
)
    @assert length(lb) == length(ub) == length(idxs) == 2

    xrange = range(lb[1], ub[1], length=ceil(Int, sample_size/4) )
    yrange = range(lb[2], ub[2], length=ceil(Int, sample_size/4) )

    Lx = length(xrange)
    Ly = length(yrange)

    perturbations = zeros(2, 2Lx+2Ly)
    for (ix, x) in enumerate(xrange)
        perturbations[:, ix]= [x, lb[2]] # bottom
        perturbations[:, Lx+ix]= [x, ub[2]] # top
    end
    for (iy, y) in enumerate(yrange)
        perturbations[:, 2Lx+iy]= [lb[1], y] # left
        perturbations[:, 2Lx+Ly+iy]= [ub[1], y] # right
    end

    if verbose
        println("Rectangle of initial conditions.")
    end

    return ICset(state, perturbations, idxs)
end
