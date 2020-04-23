const N01 = Normal(0, 1)

const REFERENCE = """
Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions
Authors: Alan Agresti and  Brent A. Coull
Publication: The American Statistician, Vol. 52, No. 2 (May, 1998), pp. 119-126
Published by: Taylor & Francis, Ltd. on behalf of the American Statistical Association
Stable URL: http://www.jstor.org/stable/2685469
DOI:10.2307/2685469.
"""


"""
    mle(counts, sample_size)

returns the maximum likelihood estimator for the success proportion

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures

$(REFERENCE)
"""
function mle_proportion(counts, sample_size)
    return counts / sample_size
end

"""
    wilson_score_proportion(counts, sample_size; α=0.05)

returns the Wilson score correction for the success proportion

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures
- `α`: target error rate

$(REFERENCE)
"""
function wilson_score_proportion(counts, sample_size; α = 0.05)
    z = quantile(N01, 1 - α / 2)
    return (counts + z^2 / 2) / (sample_size + z^2)
end

"""
    add2successes2failures_proportion(counts, sample_size)

special case of the Wilson score correction corresponding to the addition of 2 success counts and 2 failure counts

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures

$(REFERENCE)
"""
function add2successes2failures_proportion(counts, sample_size)
    return (counts + 2) / (sample_size + 4)
end

"""
    wald(counts, sample_size; α=0.05)

calculates the 100(1 - α)% Wald confidence interval for the binomial
proportion `p_hat = counts / sample_size`

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures
- `α`: target error rate

z is the (1 - α/2) quantile of a standard normal distribution, i.e.
for a 0.95 confidence interval we have α = 1 - 0.95 = 0.05 and z = 1.96.

$(REFERENCE)
"""
function wald(counts, sample_size; α = 0.05)
    z = quantile(N01, 1 - α / 2)
    p_hat = mle_proportion(counts, sample_size)
    return z * sqrt(p_hat * (1 - p_hat) / sample_size)
end

"""
    agresti_coull(counts, sample_size; α=0.05)

calculates the 100(1 - α)% Wald confidence interval for the corrected Wilson score
proportion `p_hat = (counts + z^2 / 2) / (sample_size + z^2)`

z is the (1 - α/2) quantile of a standard normal distribution, i.e.
for a 0.95 confidence interval we have α = 1 - 0.95 = 0.05 and z = 1.96.

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures
- `α`: target error rate

$(REFERENCE)
"""
function agresti_coull(counts, sample_size; α = 0.05)
    z = quantile(N01, 1 - α / 2)
    return wald(counts + z^2 / 2.0, sample_size + z^2; α = α)
end

"""
    add2successes2failures(counts, sample_size; α=0.05)

special case of Agresti/Coull confidence interval with z=2,
corresponding to the addition of 2 success counts and 2 failure counts to the sample

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures
- `α`: target error rate

$(REFERENCE)
"""
function add2successes2failures(counts, sample_size; α = 0.05)
    return wald(counts + 2, sample_size + 4; α = α)
end

"""
    wilson_score(counts, sample_size; α=0.05)

calculates the 100(1 - α)% Wilson score interval for the corrected Wilson score
proportion `p_hat = (counts + z^2 / 2) / (sample_size + z^2)`

z is the (1 - α/2) quantile of a standard normal distribution, i.e.
for a 0.95 confidence interval we have α = 1 - 0.95 = 0.05 and z = 1.96.

# Arguments
- `counts`: number of success draws in a sample
- `sample_size`: total sample size, i.e. the sum of successes and failures
- `α`: target error rate

$(REFERENCE)
"""
function wilson_score(counts, sample_size; α = 0.05)
    z = quantile(N01, 1 - α / 2)
    p_hat = mle_proportion(counts, sample_size)
    return sqrt(p_hat * (1 - p_hat) / sample_size + z^2 / (4 * sample_size^2)) *
           z / (1 + z^2 / sample_size)
end

# define interface
binomial_proportion(s::Symbol, counts, sample_size; α = 0.05) =
    binomial_proportion(Val(s), counts, sample_size; α = α)
binomial_ci(s::Symbol, counts, sample_size; α = 0.05) =
    binomial_ci(Val(s), counts, sample_size; α = α)

# Wilson score
function binomial_proportion(::Val{:wilson}, counts, sample_size; α = 0.05, verbose=false)
    if verbose
        println("Binomial proportion based on the Wilson score formula.")
    end
    wilson_score_proportion(counts, sample_size; α = α)
end
function binomial_ci(::Val{:wilson}, counts, sample_size; α = 0.05, verbose=false)
    if verbose
        println("Confidence interval based on the Wilson score formula.")
    end
    wilson_score(counts, sample_size; α = α)
end

# Agresti Coull
binomial_proportion(::Val{:ac}, counts, sample_size; α = 0.05) =
    wilson_score_proportion(counts, sample_size; α = α)
binomial_ci(::Val{:ac}, counts, sample_size; α = 0.05) =
    agresti_coull(counts, sample_size; α = α)

# add 2 successes and 2 failures
binomial_proportion(::Val{:add2}, counts, sample_size; α = 0.05) =
    add2successes2failures_proportion(counts, sample_size)
binomial_ci(::Val{:add2}, counts, sample_size; α = 0.05) =
    add2successes2failures(counts, sample_size; α = α)

# maimum likelihood + Wald interval
binomial_proportion(::Val{:mle}, counts, sample_size; α = 0.05) =
    mle_proportion(counts, sample_size)
binomial_ci(::Val{:wald}, counts, sample_size; α = 0.05) =
    wald(counts, sample_size; α = α)

# set the defaults
binomial_proportion(counts, sample_size; α = 0.05) =
    binomial_proportion(:wilson, counts, sample_size; α = α)
binomial_ci(counts, sample_size; α = 0.05) =
    binomial_ci(:wilson, counts, sample_size; α = α)

    """
        bootstrap_ci(data, statistic, m=50, α=0.05)

    provides very basic bootstrap sample of `statistic` over an array of `data`.

    # Arguments
    - `data`: univariate sample for which a `statistic` should be estimated
    - `statistic`: function that can be applied to `data` and returns a scalar statistic
    - `m`: number of bootstrap resamples
    - `α`: target error rate

    Note that bootstrapping needs some time and computation. Some testing indicates
    that it is roughly two orders of magnitude compared to `binomial_ci`.

    """
function bootstrap_ci(data, statistic::Function, m=50, α=0.05)
    n = length(data)
    empirical_dist = return [data[rand(1:n, n)] |> statistic for _ in 1:m]
    θ = statistic(data)
    lq, uq = quantile(empirical_dist, [1-α/2, α/2])
    return (θ, 2θ - lq, 2θ - uq)
end
