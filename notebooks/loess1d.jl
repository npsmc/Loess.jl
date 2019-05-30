# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

# https://gist.github.com/agramfort/850437

using Statistics, LinearAlgebra

"""
    yest = lowess(x, y, f=2./3., iter=3)

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
"""
function lowess(x, y; f = 2/3, iter=3)

    n = length(x)
    r = Int(ceil(f * n))
    h = [sort(abs.(x .- x[i]))[r] for i in eachindex(x)]
    w = clamp.(abs.((x .- x') ./ h), 0.0, 1.0)
    w .= (1 .- w.^3) .^ 3
    yest = zeros(Float64,n)
    residuals = similar(yest)
    delta = ones(Float64,n)
    weights = similar(delta)
    for iteration in 1:iter
        for i in 1:n
            weights .= delta .* view(w, :, i)
            b = [sum(weights .* y); sum(weights .* y .* x)]
            A = SymTridiagonal([sum(weights), sum(weights .* x .* x)],
                               [sum(weights .* x)])
            beta = A \ b
            yest[i] = beta[1] + beta[2] * x[i]
        end
            
        residuals .= y .- yest
        s = median(abs.(residuals))
        delta .= clamp.(residuals / (6.0 * s), -1, 1)
        delta .= (1 .- delta.^2).^2
    end
            
    yest
end
        

n = 1000
x = range(0, stop=2π, length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
f = 0.25


using Plots, BenchmarkTools
yest = lowess(x, y, f=f, iter=1);

scatter(x, y; markersize=2, label=:noisy)
plot!(x, yest; line=(3,:red),label=:pred)


