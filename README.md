<p align="center">
  <img src="logo.png" alt="Logo" width="200">
</p>

# MonteCarloJackknife.jl

MonteCarloJackknife.jl is a Julia package for efficient **Monte Carlo approximation of the delete-(d) jackknife**. Instead of exhaustively evaluating all possible delete-(d) subsamples, the package estimates jackknife quantities by randomly sampling a user-specified number of subsamples, making the method practical for large datasets and computationally expensive estimators.

## Features

* Generic implementation for arbitrary user-defined estimators.
* Supports scalar-valued and vector-valued statistics.
* Supports vectors, matrices, `DataFrame`s, and tuple-based datasets (e.g. `(X, y)`).
* Computes

  * Monte Carlo replicate estimates,
  * Monte Carlo mean,
  * bias-corrected delete-(d) jackknife estimate,
  * jackknife standard error.
* Optional multi-threading for expensive estimation functions.

## Installation

```julia
using Pkg
Pkg.add("MonteCarloJackknife")
```

## API

```julia
mc_delete_d_jackknife(
    f::Function,
    data,
    d::Int;
    num_samples = 1000,
    multithreaded = false
)
```

### Arguments

| Argument        | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `f`             | User-defined estimator or statistic.                                     |
| `data`          | Dataset supplied to `f`. May be a vector, matrix, `DataFrame`, or tuple. |
| `d`             | Number of observations deleted in each subsample.                        |
| `num_samples`   | Number of Monte Carlo delete-(d) subsamples.                             |
| `multithreaded` | Whether to evaluate subsamples in parallel.                              |

## A Simple Univariate Example (Median)

Estimate the sampling variability of the sample median using Monte Carlo delete-(d) jackknife.

```julia
using MonteCarloJackknife
using Random

Random.seed!(1234)

x = randn(1000)

result = mc_delete_d_jackknife(
    median,
    x,
    200;
    num_samples = 1000
)

println(result.mean_jackknife)
println(result.std_error)
```

The returned object contains

* `replicates` — Monte Carlo replicate estimates.
* `mean_jackknife` — Mean of the Monte Carlo replicates.
* `bias_corrected_mean` — Bias-corrected delete-(d) jackknife estimate.
* `std_error` — Delete-(d) jackknife standard error.

## A Multivariate Example (Regression)

The package also supports vector-valued estimators.

```julia
using MonteCarloJackknife
using Random
using LinearAlgebra

Random.seed!(1234)

# Generate synthetic regression data
n = 1000
x1 = randn(n)
x2 = randn(n)

X = hcat(ones(n), x1, x2)

β = [2.0, 1.5, -0.8]
y = X * β + randn(n)

# Ordinary least squares estimator
function ols(data)
    X, y = data
    return X \ y
end

result = mc_delete_d_jackknife(
    ols,
    (X, y),
    200;
    num_samples = 1000
)

println("Coefficient estimates:")
println(result.mean_jackknife)

println("\nStandard errors:")
println(result.std_error)
```

In this example, `result.replicates` matrix contains one row for each regression coefficient and one column for each Monte Carlo replicate.


## Applications

MonteCarloJackknife.jl can be used for:

* Computationally efficient approximation of delete-(d) jackknife for large datasets.
* Standard error estimation.
* Bias estimation and bias correction.
* Assessing estimator stability and multicollinearity.
* Statistical inference for custom estimators.
* Fitting machine learning models on subsamples to assess feature importance and parameter sensitivity to sample size.

## Examples

Additional and more detailed examples are available in the `examples/` directory, including:

* Sample median
* Pearson correlation
* Ordinary least squares regression

## Citation

Coming soon...

## Acknowledgements

The main algorithm that I implemented in this package was inspired by one of the research topics that I had tackled during my PhD studies. For this reason,
I would like to thank my former academic advisors İlker Birbil ([@sibirbil](https://github.com/sibirbil)) and Sinan Yıldırım ([@sinanyildirim](https://github.com/sinanyildirim)) for the valuable hours that we had spent together working on that topic.

## License

This package is released under the MIT License.

