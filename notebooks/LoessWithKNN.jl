# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

using LinearAlgebra, Distributions, Statistics, Plots
using NearestNeighbors

#Initializing noisy non linear data
x = range(0,stop=1,length=100) |> collect
y = sin.(x .* 3π/2 )
σ = 0.1
d = MvNormal([0], σ .* Matrix(I,1,1))
n = length(x)
y_noise = y .+ rand(d,n)';

scatter( x, y_noise)
plot!(x, y)

# +
"""
    lowess_knn(x, y, tau = .005) -> yest

Locally weighted regression: fits a nonparametric regression curve 
to a scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function 
returns the estimated (smooth) values of y.
The kernel function is the bell shaped function with parameter tau. 
Larger tau will result in a smoother curve. 
"""
function lowess(x, k = 3)
    
    m = last(size(x))
    kdt = KDTree(x; leafsize = 10)
    idxs, dists = knn(kdt, x, k, true)
    yest = zeros(Float64, m)
    
    w = (1 .- dists.^3) .^ 3
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


