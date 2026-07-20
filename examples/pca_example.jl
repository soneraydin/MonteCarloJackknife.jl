# This example is almost the same as the one in the README file, but here we also add a scree plot
using MonteCarloJackknife
using MultivariateStats
using Random
using Plots

Random.seed!(1234)

# Generate synthetic data
n = 1000
p = 100
k = 10 # Number of latent dimensions

Z = randn(n, k)
W = randn(k, p)

X = Z * W + 0.2 * randn(n, p)

# PCA estimator returning ordered explained variance ratios
function pca_scree(data)
    model = fit(PCA, data; maxoutdim=size(data, 2))
    principalvars(model) ./ tvar(model)
end

result = mc_delete_d_jackknife(
    pca_scree,
    X,
    200;
    num_samples = 1000
)

println(result.mean_jackknife)
println(result.std_error)

plot(
    result.mean_jackknife;
    ribbon=1.95*result.std_error,
    xlabel="Principal Component",
    ylabel="Explained Variance Ratio",
    label="Estimate ± 1.95 SE",
    legend=:topright
)
