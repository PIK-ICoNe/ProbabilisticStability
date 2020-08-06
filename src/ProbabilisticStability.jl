module ProbabilisticStability

folder = dirname(@__FILE__)

using Distributions: Normal, quantile
include(folder * "/SampleStatistics.jl")
export binomial_proportion, binomial_ci

using QuasiMonteCarlo
import Base: size, length, getindex # extend these functions
include(folder * "/InitialConditionSets.jl")
export perturbation_set,
    perturbation_set_uniform, perturbation_set_sobol, perturbation_set_grid, perturbation_set_rect
export ICset

using Distributions: mean
using Distances: euclidean, Euclidean, PeriodicEuclidean, colwise, evaluate
using DiffEqBase: AbstractODESolution
include(folder * "/Observables.jl")
export get_max_distance_to_state,
    get_max_distances_to_state,
    get_mean_distance_to_state,
    get_trajectory_within_bounds,
    get_convergence_to_state,
    eval_convergence_to_state,
    eval_distance_to_state,
    eval_final_distance_to_state,
    eval_max_distance_to_state,
    eval_max_distances_to_state,
    eval_mean_distance_to_state,
    eval_trajectory_within_bounds

using DynamicalSystems:
    DynamicalSystem, parallel_integrator, SVector, trajectory, step!, get_state
using DiffEqCallbacks: TerminateSteadyState
using OrdinaryDiffEq:
    EnsembleProblem,
    ODEProblem,
    AutoTsit5,
    Rosenbrock23,
    Rodas4,
    solve,
    EnsembleThreads,
    EnsembleSerial,
    remake
using Distributed: nprocs
using DataFrames: DataFrame, rename!
include(folder * "/Sampling.jl")
export basin_stability_fixpoint
export survivability

using Plots: plot, scatter!, xlims!, ylims!, plot!
using DataFrames: DataFrame
include(folder * "/Plots.jl")
export basin_plot_trajectories
export basin_plot


end #module
