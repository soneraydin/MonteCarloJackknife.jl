using MonteCarloJackknife, Random, Statistics, Distributions, Plots

Random.seed!(1234)

x = rand(Exponential(0.5), 2000)

d = round(Int, 0.33*length(x))

median_fun(x) = median(x)

result = mc_delete_d_jackknife(
    median_fun,
    x,
    d;
    num_samples = 1000
)

thetas = result.replicates
std_err = result.std_error
full_theta = median_fun(x)

# Empirical interval of replicate distribution
low = quantile(thetas, 0.025)
high = quantile(thetas, 0.975)

histogram(
    thetas,
    bins=:auto,
    xlabel="Median",
    ylabel="Frequency",
    title="Monte Carlo Delete-d Jackknife Replicates",
    label="Replicates",
    alpha=0.7,
    color=:skyblue
)

vline!(
    [full_theta],
    linewidth=2,
    color=:red,
    label="Full Sample Median"
)

vline!(
    [low,high],
    linestyle=:dash,
    linewidth=2,
    color=:purple,
    label="95% Replicate Interval"
)