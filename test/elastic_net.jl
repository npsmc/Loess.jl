using Test
using LinearAlgebra
using ProgressMeter
using Random
using StatsPlots
using Statistics
using GLMNet
using ScikitLearn

@testset "Elastic Net" begin

@sk_import linear_model: (lars_path, ElasticNetCV)

include("DataGeneration.jl")

function rond(x, s=3)
    nx = s - 1 - floor(Int64, log10(abs(x) + 1.e-32))
    round(x, digits=nx)
end

rondv(x, s=3) = [rond(xx, s) for (i, xx) in enumerate(x)]
nor2(x) = sum(x.^2)
nor(x) = sqrt(nor2(x))

    Random.seed!(seed)
    
    X, y, betas = RandomLM(ny=200, p=400, sy=1, nbet=30, mbet=4,
                           sdbet=0, Xbr=-0.05)
    
    cv = glmnetcv(X, y; intercept=false)
    best   = argmin(cv.meanloss)
    betnet = cv.path.betas[:, best]
    err = nor2(X * (betnet - betas)) / nor2(X * betas)

@test err < 1e-3

end
