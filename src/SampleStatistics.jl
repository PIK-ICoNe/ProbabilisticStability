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
    empirical_dist = [data[rand(1:n, n)] |> statistic for _ in 1:m]
    θ = statistic(data)
    #lq, uq = quantile(empirical_dist, [1-α/2, α/2])
    #2θ - lq, 2θ - uq
    return θ, std(empirical_dist)
end

# define interface
sample_statistics(s::Symbol, observations; α = 0.05) = sample_statistics(Val(s), observations; α = α)


# Wilson score
function sample_statistics(::Val{:wilson}, observations; α = 0.05, verbose=false)
    if verbose
        println("Binomial confidence interval based on the Wilson score formula.")
    end
    counts = count(observations)
    sample_size = length(observations)
    return wilson_score_proportion(counts, sample_size; α = α), wilson_score(counts, sample_size; α = α), counts
end


# Agresti Coull
function sample_statistics(::Val{:ac}, observations; α = 0.05, verbose=false)
    if verbose
        println("Agresti&Coull confidence interval.")
    end
    counts = count(observations)
    sample_size = length(observations)
    return wilson_score_proportion(counts, sample_size; α = α), agresti_coull(counts, sample_size; α = α), counts
end


# add 2 successes and 2 failures
function sample_statistics(::Val{:add2}, observations; α = 0.05, verbose=false)
    if verbose
        println("Binomial confidence interval based by adding 2 successes and 2 failures.")
    end
    counts = count(observations)
    sample_size = length(observations)
    return add2successes2failures_proportion(counts, sample_size), add2successes2failures(counts, sample_size; α = α), counts
end

# maximum likelihood + Wald interval
function sample_statistics(::Val{:mle}, observations; α = 0.05, verbose=false)
    if verbose
        println("Binomial confidence interval based by adding 2 successes and 2 failures.")
    end
    counts = count(observations)
    sample_size = length(observations)
    return mle_proportion(counts, sample_size), wald(counts, sample_size; α = α), counts
end

# bootstrapping
function sample_statistics(::Val{:bootstrap}, observations; α = 0.05, verbose=false)
    if verbose
        println("Binomial confidence interval based by adding 2 successes and 2 failures.")
    end
    counts = count(observations)
    return bootstrap_ci(observations, mean, 100, α)..., counts
end


# set the defaults
 sample_statistics(observations; α = 0.05) =
     sample_statistics(:wilson, observations; α = α)
