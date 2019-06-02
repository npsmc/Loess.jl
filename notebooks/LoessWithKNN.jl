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

using LinearAlgebra, Distributions, Statistics, Plots
using NearestNeighbors

#Initializing noisy non linear data
x = range(0,stop=1,length=100) |> collect
y = hcat([sin(t * 3π/2 ) for t in x],
         [cos(t * 3π/2 ) for t in x])' |> collect
σ = 0.1
d = MvNormal([0,0], σ .* Matrix(I,2,2))
n = length(x)
y_noise = y .+ rand(d,n)

scatter( x, y_noise[1,:])
scatter!(x, y_noise[2,:])
plot!(x, y[1,:])
plot!(x, y[2,:])

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
function lowess_bell_shape_kern(x, k = 3)
    
    m = last(size(x))
    kdt = KDTree(x; leafsize = 10)
    idxs, dists = knn(kdt, x, k, true)
    yest = zeros(Float64, m)
    
    w = (1 .- w.^3) .^ 3
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
