using MonteCarloJackknife
using Test
using Random
using Statistics
using LinearAlgebra
using DataFrames

Random.seed!(1234)

@testset "MonteCarloJackknife.jl" begin

    ########################################################################
    # Test 1: Scalar-valued statistic (median)
    ########################################################################

    @testset "Median" begin

        x = randn(500)

        result = mc_delete_d_jackknife(
            median,
            x,
            100;
            num_samples = 250
        )

        @test length(result.replicates) == 250
        @test isa(result.mean_jackknife, Float64)
        @test isa(result.std_error, Float64)
        @test result.std_error > 0
        @test isfinite(result.bias_corrected_mean)

    end

    ########################################################################
    # Test 2: Vector-valued statistic (multiple regression)
    ########################################################################

    @testset "OLS regression" begin

        n = 500

        x1 = randn(n)
        x2 = randn(n)
        x3 = 0.9 .* x1 .+ sqrt(1 - 0.9^2) .* randn(n)

        X = hcat(ones(n), x1, x2, x3)

        βtrue = [2.0, 3.0, 0.0, 1.5]

        y = X * βtrue + randn(n)

        dataset = (X, y)

        ols(data) = begin
            X, y = data
            X \ y
        end

        result = mc_delete_d_jackknife(
            ols,
            dataset,
            150;
            num_samples = 200
        )

        @test size(result.replicates) == (4, 200)
        @test length(result.mean_jackknife) == 4
        @test length(result.std_error) == 4
        @test all(result.std_error .> 0)

    end

    ########################################################################
    # Test 3: DataFrame input
    ########################################################################

    @testset "DataFrame" begin

        n = 400

        df = DataFrame(
            x = randn(n),
            y = randn(n)
        )

        corrfun(df) = cor(df.x, df.y)

        result = mc_delete_d_jackknife(
            corrfun,
            df,
            100;
            num_samples = 200
        )

        @test length(result.replicates) == 200
        @test abs(result.mean_jackknife) ≤ 1
        @test result.std_error > 0

    end

end