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

# https://xavierbourretsicotte.github.io/loess.html

using LinearAlgebra, Distributions, Statistics, Plots

# +
"""
    lowess_bell_shape_kern(x, y, tau = .005) -> yest

Locally weighted regression: fits a nonparametric regression curve 
to a scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function 
returns the estimated (smooth) values of y.
The kernel function is the bell shaped function with parameter tau. 
Larger tau will result in a smoother curve. 
"""
function lowess_bell_shape_kern(x, y, tau = 0.005)
    
    m = length(x)
    yest = zeros(Float64, m)
    
    #Initializing all weights from the bell shape kernel function    
    w = [exp.(- (x .- x[i]).^2 ./(2*tau)) for i in 1:m] 
    weights = zeros(Float64, m)
    
    #Looping through all x-points
    for i in 1:m
        weights .= w[i]
        b = [sum(weights .* y), sum(weights .* y .* x)]
        A = SymTridiagonal(
                [sum(weights), sum(weights .* x .* x)],
                [sum(weights .* x)])
        theta = A \ b
        yest[i] = theta[1] + theta[2] * x[i]
    end

    yest
    
end
# -

"""
    lowess(x, y, f=2./3., iter=3) -> yest

Lowess smoother: Robust locally weighted regression.
The lowess function fits a nonparametric regression curve to a 
scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function 
returns
the estimated (smooth) values of y.
The smoothing span is given by f. A larger value for f will result 
in a smoother curve. The number of robustifying iterations is 
given by iter. The function will run faster with a smaller number 
of iterations.
"""
function lowess_ag(x, y; f=2/3, iter=3)
    
    n = length(x)
    r = Int(ceil(f * n))
    h = [sort(abs.(x .- x[i]))[r] for i in 1:n]
    w = clamp.(abs.((x .- x') ./ h), 0.0, 1.0)
    w = (1 .- w.^3) .^ 3
    yest = zeros(n)
    delta = ones(n)
    for iteration in 1:iter
        for i in 1:n
            weights = delta .* w[:, i]
            b = [sum(weights .* y); sum(weights .* y .* x)]
            A = SymTridiagonal([sum(weights), sum(weights .* x .* x)],
                          [sum(weights .* x)])
            beta = A \ b
            yest[i] = beta[1] + beta[2] * x[i]
        end

        residuals = y .- yest
        s = median(abs.(residuals))
        delta = clamp.(residuals ./ (6.0 * s), -1, 1)
        delta = (1 .- delta .^ 2) .^  2
    end

    yest
end

#Initializing noisy non linear data
x = range(0,stop=1,length=100) |> collect
noise = 0.2 * randn(100)
y = sin.(x * 3π/2 ) 
y_noise = y .+ noise
f = 0.25
yest = lowess_ag(x, y_noise; f=f, iter=3)
yest_bell = lowess_bell_shape_kern(x,y_noise)
plot(x, yest; label="gramfort")
plot!(x, yest_bell; label="bellshape")
scatter!(x, y_noise)


