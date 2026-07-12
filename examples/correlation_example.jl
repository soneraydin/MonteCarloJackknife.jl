using MonteCarloJackknife, Random, Statistics, Plots

# Random.seed!(1234)
x = randn(2000) .+ 5
y = 3*x - 0.2*x.^2 + randn(2000)
dataset = hcat(x, y)

# d = Int(round(sqrt(size(dataset)[1])) + 1)
d = Int(round(0.33*size(dataset)[1]))

function corr_func(dataset)
    return cor(dataset[:, 1], dataset[:, 2])
end

full_sample_theta = corr_func(dataset)

# Run cleaner execution script without keywords for random configurations
result = mc_delete_d_jackknife(corr_func, dataset, d; num_samples=1000)

thetas = result.replicates
mu_jack = result.mean_jackknife
std_err = result.std_error
bc_mu = result.bias_corrected_mean

println("Estimated Mean of given function: ", mu_jack)
println("MC-Jackknife Standard Error: ", std_err)

# 1. Compute the empirical 95% confidence intervals
ci_low = quantile(thetas, 0.025)
ci_high = quantile(thetas, 0.975)

# 2. Generate the base histogram
histogram(thetas, 
    bins = :auto, 
    label = "MC Jackknife Estimates", 
    title = "Distribution of MC Jackknife Thetas", 
    xlabel = "Value of Statistic", 
    ylabel = "Frequency", 
    color = :skyblue, 
    alpha = 0.7, 
    edgecolor = :white,
    legendfontsize = 7
)

# 3. Add the vertical mean line
vline!([full_sample_theta], color = :red, linewidth = 2, label = "Full Sample Correlation")

# 4. Add the 95% CI lines (darkblue and dashed)
vline!([ci_low, ci_high], 
    color = :purple, 
    linestyle = :dash, 
    linewidth = 2, 
    label = "95% Monte Carlo Replicate Interval"
)