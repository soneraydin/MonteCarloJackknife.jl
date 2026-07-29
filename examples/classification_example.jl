using Random
using Statistics
using LinearAlgebra
using DataFrames
using GLM
using MonteCarloJackknife
using StatsPlots

Random.seed!(1234)

# Simulate binary logistic regression data

n = 2000

# Latent factors
z1 = randn(n)
z2 = -0.6 .* z1 .+ sqrt(1 - 0.6^2) .* randn(n)
z3 = -0.6 .* z2 .+ sqrt(1 - 0.6^2) .* randn(n)

# Within-group correlation
ρ = 0.8

x1 =  z1 .+ sqrt(1-ρ^2) .* randn(n)
x2 =  ρ .* z1 .+ sqrt(1-ρ^2) .* randn(n)

x3 =  z2 .+ sqrt(1-ρ^2) .* randn(n)
x4 =  ρ .* z2 .+ sqrt(1-ρ^2) .* randn(n)

x5 =  z3 .+ sqrt(1-ρ^2) .* randn(n)
x6 =  ρ .* z3 .+ sqrt(1-ρ^2) .* randn(n)

βtrue = [
    -0.5,   # intercept
     2.0,
     1.5,
     0.0,
    -1.5,
     0,
    -2.0
]

η = βtrue[1] .+
    βtrue[2] .* x1 .+
    βtrue[3] .* x2 .+
    βtrue[4] .* x3 .+
    βtrue[5] .* x4 .+
    βtrue[6] .* x5 .+
    βtrue[7] .* x6

p = 1 ./ (1 .+ exp.(-η))

y = Int.(rand(n) .< p)

dataset = DataFrame(
    y = y,
    x1 = x1,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6
)

# Logistic regression function

function logistic_regression(data)
    model = glm(
        @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6),
        data,
        Binomial(),
        LogitLink()
    )
    return coef(model)
end

full_sample = logistic_regression(dataset)

# Monte Carlo delete-d jackknife

d = round(Int, 0.8n)

result = mc_delete_d_jackknife(
    logistic_regression,
    dataset,
    d;
    num_samples = 1000
)

println("Full-sample estimates:")
println(full_sample)

println("\nBias-corrected estimates:")
println(result.bias_corrected_mean)

println("\nMonte Carlo Jackknife standard errors:")
println(result.std_error)

# Visualization

coef_names = ["β₀","β₁","β₂","β₃","β₄","β₅","β₆"]

# Each column corresponds to one parameter
betas = result.replicates'

# Boxplot with true values

p1 = boxplot(
    betas;
    xticks = (1:7, coef_names),
    xlabel = "Coefficient",
    ylabel = "Estimate",
    title = "Monte Carlo Jackknife Estimates",
    label = "",
    legend = :topright,
    linewidth = 2,
    fillalpha = 0.5,
    outliers = true,
    size = (700,500)
)
scatter!(
    p1,
    1:length(βtrue),
    βtrue;
    marker = (:star5, 12),
    color = :red,
    markerstrokecolor = :black,
    label = "True value"
)
