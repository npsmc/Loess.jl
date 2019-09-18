# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

# https://xavierbourretsicotte.github.io/loess.html

using LinearAlgebra, Statistics, Plots

# +
"""
    lowess_1d(x, y, tau = .005) -> yest

Locally weighted regression: fits a nonparametric regression curve 
to a scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function 
returns the estimated (smooth) values of y.
The kernel function is the bell shaped function with parameter tau. 
Larger tau will result in a smoother curve. 
"""
function lowess_1d(x, y; tau = 0.005)
    
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

#Initializing noisy non linear data
x = range(0,stop=1,length=100) |> collect
noise = 0.2 * randn(100)
y = sin.(x * 3π/2 ) 
y_noise = y .+ noise
ypred = lowess_1d(x, y_noise; tau = 1e-2)
plot(x, ypred; label="lowess")
scatter!(x, y_noise)


