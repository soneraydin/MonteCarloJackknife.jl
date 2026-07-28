using Random
using Statistics
using LinearAlgebra
using BenchmarkTools
using DataFrames
using MultivariateStats
using MonteCarloJackknife

############################################################
# Benchmark Settings
############################################################

Random.seed!(1234)

NUM_SAMPLES = 200          # Monte Carlo subsets
DELETE_RATIO = 0.20

println("==========================================")
println("Julia Version : ", VERSION)
println("CPU Threads   : ", Threads.nthreads())
println("==========================================")

############################################################
# Result table
############################################################

results = DataFrame(
    Problem = String[],
    DataDimensions = String[],
    Serial_ms = Float64[],
    Parallel_ms = Float64[],
    Speedup = Float64[]
)

############################################################
# Benchmark helper
############################################################

function benchmark_problem(problem_name,
                           dims,
                           estimator,
                           data,
                           d)

    # Warm-up (JIT compilation)
    mc_delete_d_jackknife(
        estimator,
        data,
        d;
        num_samples=NUM_SAMPLES,
        multithreaded=false
    )

    mc_delete_d_jackknife(
        estimator,
        data,
        d;
        num_samples=NUM_SAMPLES,
        multithreaded=true
    )

    println("Benchmarking: ", problem_name)

    serial = @belapsed mc_delete_d_jackknife(
        $estimator,
        $data,
        $d;
        num_samples=$NUM_SAMPLES,
        multithreaded=false
    )

    parallel = @belapsed mc_delete_d_jackknife(
        $estimator,
        $data,
        $d;
        num_samples=$NUM_SAMPLES,
        multithreaded=true
    )

    push!(
        results,
        (
            problem_name,
            dims,
            serial * 1000,
            parallel * 1000,
            serial / parallel
        )
    )

end

############################################################
# Example 1
# Multiple Linear Regression
############################################################

println()
println("Generating regression data...")

n = 1000
p = 100

X = randn(n, p)

β = randn(p)

y = X * β .+ randn(n)

function regression_estimator(data)

    X, y = data

    X \ y

end

d = round(Int, DELETE_RATIO * n)

benchmark_problem(
    "Multiple Linear Regression",
    "1000 × 100",
    regression_estimator,
    (X, y),
    d
)

############################################################
# Example 2
# Binary Classification
############################################################

println()
println("Generating classification data...")

n = 1000
p = 100

X = randn(n, p)

β = randn(p)

η = X * β

prob = 1 ./ (1 .+ exp.(-η))

y = prob .> rand(n)

"""
A lightweight estimator.

Fits a linear classifier using least squares and returns
classification accuracy.
"""
function classification_estimator(data)

    X, y = data

    β̂ = X \ Float64.(y)

    p̂ = 1 ./(1 .+ exp.(-X * β̂))

    mean((p̂ .> 0.5) .== y)

end

d = round(Int, DELETE_RATIO * n)

benchmark_problem(
    "Binary Classification",
    "1000 × 100",
    classification_estimator,
    (X, y),
    d
)

############################################################
# Example 3
# PCA
############################################################

println()
println("Generating PCA data...")

n = 1000
p = 100
latent_dim = 10

Z = randn(n, latent_dim)

W = randn(latent_dim, p)

X = Z * W .+ 0.25 .* randn(n, p)

function pca_estimator(X)

    model = fit(PCA, X)

    principalvars(model) ./ tvar(model)

end

d = round(Int, DELETE_RATIO * n)

benchmark_problem(
    "Principal Component Analysis",
    "1000 × 100",
    pca_estimator,
    X,
    d
)

############################################################
# Results
############################################################

println()
println("==========================================")
println("Benchmark Results")
println("==========================================")
println()

show(results, allrows=true, allcols=true)

println()
