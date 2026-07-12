using MonteCarloJackknife, Random, Statistics, LinearAlgebra, Plots

Random.seed!(1234)

# -------------------------------------------------------------------------
# Simulate regression data
# -------------------------------------------------------------------------

n = 2000

ρ13 = 0.8      # Strong correlation between x1 and x3
ρ12 = -0.5      # Weak correlation between x1 and x2

x1 = randn(n)

x2 = ρ12 .* x1 .+ sqrt(1 - ρ12^2) .* randn(n)

x3 = ρ13 .* x1 .+ sqrt(1 - ρ13^2) .* randn(n)

X = hcat(ones(n), x1, x2, x3)

βtrue = [2.0, 3.0, 0.0, 1.5]

y = X * βtrue + randn(n)

dataset = (X, y)

function ols(data)
    X, y = data
    return X \ y
end

beta_hat = X \ y

# -------------------------------------------------------------------------
# Monte Carlo delete-d jackknife
# -------------------------------------------------------------------------

d = round(Int, 0.9*n)

result = mc_delete_d_jackknife(
    ols,
    dataset,
    d;
    num_samples = 1000
)

betas = result.replicates

println("Full-sample estimates:")
println(beta_hat)

println("\nMonte Carlo Jackknife standard errors:")
println(result.std_error)

# -------------------------------------------------------------------------
# Histograms of coefficient estimates
# -------------------------------------------------------------------------

bc_beta = result.bias_corrected_mean

lower = [quantile(betas[i,:], 0.025) for i in 1:4]
upper = [quantile(betas[i,:], 0.975) for i in 1:4]

labels = ["Intercept", "β₁", "β₂", "β₃"]

p = plot(layout=(2,2), size=(900,700))

for i in 1:4

    histogram!(
        p[i],
        betas[i,:],
        bins=:auto,
        title=labels[i],
        xlabel="Estimate",
        ylabel="Frequency",
        color=:skyblue,
        alpha=0.7,
        label="MC replicates"
    )

    # Bias-corrected estimate
    vline!(
        p[i],
        [bc_beta[i]],
        color=:black,
        linewidth=2,
        label="Bias-corrected"
    )

    # Empirical 95% interval
    vline!(
        p[i],
        [lower[i], upper[i]],
        color=:forestgreen,
        linestyle=:dot,
        linewidth=2,
        label="95% empirical interval"
    )

    # True coefficient
    vline!(
        p[i],
        [βtrue[i]],
        color=:red,
        linestyle=:dash,
        linewidth=2,
        label="True value"
    )

end

display(p)

# -------------------------------------------------------------------------
# Correlation heatmap
# -------------------------------------------------------------------------

corrmat = cor(permutedims(betas))

heatmap(
    corrmat,
    xticks = (1:4, labels),
    yticks = (1:4, labels),
    aspect_ratio = :equal,
    c = :RdBu,
    clims = (-1,1),
    title = "Correlation Among Monte Carlo Jackknife Estimates",
    colorbar_title = "Correlation"
)
