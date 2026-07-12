module MonteCarloJackknife

using Statistics
using Random
using StatsBase
using DataFrames
using Base.Threads

export mc_delete_d_jackknife

"""
    mc_delete_d_jackknife(f::Function, data, d::Int;
                          num_samples::Int=1000,
                          multithreaded=false)

Perform a Monte Carlo delete-d jackknife on `data` for any function `f`,
applying the mathematically corrected scaling factor for the delete-d
jackknife standard error.
"""
function mc_delete_d_jackknife(
    f::Function,
    data,
    d::Int;
    num_samples::Int=1000,
    multithreaded=false
)

    # ----------------------------------------------------------------------
    # Determine sample size
    # ----------------------------------------------------------------------
    n = nobs(data)
    r = n - d

    r > 0 || throw(ArgumentError("d ($d) must be less than sample size n ($n)"))

    # ----------------------------------------------------------------------
    # Determine output type
    # ----------------------------------------------------------------------
    test_idx = sample(1:n, r; replace=false)
    first_eval = f(select_obs(data, test_idx))
    results_type = typeof(first_eval)

    # ----------------------------------------------------------------------
    # Allocate storage
    # ----------------------------------------------------------------------
    if results_type <: Number
        thetas = Vector{Float64}(undef, num_samples)
    else
        thetas = Matrix{Float64}(undef, length(first_eval), num_samples)
    end

    # ----------------------------------------------------------------------
    # Monte Carlo sampling
    # ----------------------------------------------------------------------
    if multithreaded

        Threads.@threads for i in 1:num_samples

            idx = sample(1:n, r; replace=false)
            sub_sample = select_obs(data, idx)

            if results_type <: Number
                thetas[i] = Float64(f(sub_sample))
            else
                thetas[:, i] = Float64.(f(sub_sample))
            end

        end

    else

        for i in 1:num_samples

            idx = sample(1:n, r; replace=false)
            sub_sample = select_obs(data, idx)

            if results_type <: Number
                thetas[i] = Float64(f(sub_sample))
            else
                thetas[:, i] = Float64.(f(sub_sample))
            end

        end

    end

    # ----------------------------------------------------------------------
    # Compute jackknife estimates
    # ----------------------------------------------------------------------
    theta_full = f(data)

    if results_type <: Number
        mean_theta = mean(thetas)
    else
        mean_theta = vec(mean(thetas, dims=2))
    end

    bias_corrected_theta =
        (n .* theta_full .- (n - d) .* mean_theta) ./ d

    # ----------------------------------------------------------------------
    # Standard error
    # ----------------------------------------------------------------------
    if results_type <: Number
        raw_std = std(thetas; corrected=false)
    else
        raw_std = vec(std(thetas, dims=2; corrected=false))
    end

    correction_factor = sqrt((n - d) / d)
    corrected_std_theta = raw_std .* correction_factor

    return (
        replicates = thetas,
        mean_jackknife = mean_theta,
        std_error = corrected_std_theta,
        bias_corrected_mean = bias_corrected_theta
    )

end


# ==========================================================================
# Dataset interface
# ==========================================================================

"""
    nobs(data)

Return the number of observations in `data`.
"""

function nobs(data::AbstractVector)
    return length(data)
end

function nobs(data::AbstractMatrix)
    return size(data, 1)
end

function nobs(data::DataFrame)
    return nrow(data)
end

function nobs(data::Tuple)

    n = nobs(first(data))

    for x in Base.tail(data)

        nobs(x) == n ||
            throw(ArgumentError(
                "All tuple elements must have the same number of observations."
            ))

    end

    return n

end


"""
    select_obs(data, idx)

Return observations indexed by `idx`.
"""

function select_obs(data::AbstractVector, idx)
    return data[idx]
end

function select_obs(data::AbstractMatrix, idx)
    return data[idx, :]
end

function select_obs(data::DataFrame, idx)
    return @view data[idx, :]
end

function select_obs(data::Tuple, idx)
    return map(x -> select_obs(x, idx), data)
end

end # module