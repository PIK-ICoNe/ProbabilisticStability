# ProbabilisticStability
An implementation of differnet probabilistic stability measures for deterministic systems.

The aim is to combine features from different Julia packages to implement stability analysis 
methods developed in recent papaers, e.g. from PIK.

- NetworkDynamics.jl to represent networked systems efficiently
- DynamicalSystems.jl as an alternative interface primarily for low-dimensional systems, use of parallel integrator
- MCBB.jl/Clustering.jl for Monte-Carlo sampling and trajectory clustering
- QuasiMonteCarlo.jl to generate low-discrepancy sequences
- DifferentialEquations.jl for numerical integration (underlying most above-mentioned packages)

As a start, first implementations are included in the `examples` folder.

The `ProbabilisticStability` package is constituted by four sub-modules.

## InitialConditionSets

Provides functions to generate random, low-discrepancy or gridded sets of initial conditions.
Since in many cases, initial conditions arise from perturbations applied to only a few dimensions, 
we can think of the perturbations as a small "patch" to the original state. Hence, initial condition 
sets are stored as ICset types:

```
struct ICset{T}
    state::Array{T}
    perturbations::Array{T}
    idxs
    outdim
end
```
Via overloading `getindex`, they can be accessed like common Array types and return the initial condition patched at the right indexes `idxs`.

TODO:

- develop per-dimension perturbations further as a base for single-node perturbations
- develop methods to sample valid initial conditions for DAEs

## Observables

Provides convenience functions for observing various distance measures and to check whether trajectories remain within given bounds. So far, only the distance between trajectories and (fix-)points is implemented.

TODO:

- measure the distance/convergence to extended attractors
- implement time-dependent observables (e.g. for FTBS)

## Sampling Routines

Provides specialised sampling routines for various probabilistic stability measures.
Implemented so far: basin stability of fixpoints.

The stability evaluation is based on Monte-Carlo sampling, currently implemented 
via the `EnsembleProblem`of DifferentialEquations.jl. In the future, this task should 
integrate MCBB, especially when parameters are varied. 

An alternative Implementation is available that uses the perallel integrator provided by
DynamicalSystems.jl. For a small system, all initial conditions are integrated at once.
Though it might be more performant than ensemble simulations in theory, first test indicate
that the parallel approach is an order of magnitude slower, probably because it cannot make 
use of adaptive step sizing.

TODO:

- add survivability, FTBS, n-node basin stability, stochastic basin stability, ...
- add parameter-varying analyses to study bifurcations 
- extend MCBB integration
    - how to set up a MCBB problem with fixed parameter
    - need function that returns the __count__ of a condition per cluster per sliding window

## SampleStatistics

Provides various statistics that estimate the mean and its confidence interval for Bernoulli experiments.
As a default, the Wilson score formula described in the Agresti&Coull paper is used.
So far, I did not find this elsewhere.

## PhaseSpacePortraits

Not implemented yet. Idea: use heatmaps to plot sampling results. Might be necessary to interpolate irregularly sampled
initial conditions.

TODO:

- use the __whole__ trajectories to plot basins.


