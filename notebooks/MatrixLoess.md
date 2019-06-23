---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.6
  kernelspec:
    display_name: Julia 1.1.1
    language: julia
    name: julia-1.1
---

```julia
using Plots, NearestNeighbors, LinearAlgebra
```

```julia
function tricubic(x)
    y = similar(x)
    fill!(y, 0.0)
    idx = (x .>= -1) .& (x .<= 1)
    y[idx] = (1.0 .- abs.(x[idx]).^3).^3
    y
end
```

```julia
x = range(-2, stop=2, length=100)
plot(x, tricubic(x); label=:tricubic)
```

```julia
function normalize_array!(array)
    min_val = minimum(array)
    max_val = maximum(array)
    array .= (array .- min_val) ./ (max_val - min_val)
    min_val, max_val
end
```

```julia
struct Loess
    
    n_xx :: Array{Float64,1}
    n_yy :: Array{Float64,1}
    degree :: Int
    min_xx :: Float64
    max_xx :: Float64
    min_yy :: Float64
    max_yy :: Float64
    kdtree :: KDTree
    
    function Loess(n_xx, n_yy, degree=1)
        min_xx, max_xx = normalize_array!(n_xx)
        min_yy, max_yy = normalize_array!(n_yy)
        kdtree = KDTree(reshape(n_xx,1,length(n_xx)))
        new(n_xx, n_yy, degree, 
            min_xx, max_xx, min_yy, max_yy,
            kdtree)
    end
    
end
```

```julia
function normalize_x(loess, value)
    (value - loess.min_xx) / (loess.max_xx - loess.min_xx)
end

function denormalize_y(loess, value)
    value * (loess.max_yy - loess.min_yy) + loess.min_yy
end
```

```julia
function estimate(loess :: Loess, x :: Float64; window=3,
        use_matrix=false, 
        degree=1)
    
    n_x = normalize_x(loess, x)
    
    max_distance = maximum(abs.(n_x .- loess.n_xx))
    @show idxs, dists = knn(loess.kdtree, [x], window, true)
    weights = tricubic(dists ./ max_distance)
    weights .= weights ./ sum(weights)

    if use_matrix || degree > 1
        wm = Diagonal(weights)
        xm = ones((window, degree + 1))

        xp = [n_x.^p for p in 0:degree]
        for i in 1:degree
            xm[:, i] = loess.n_xx[idxs]^i
        end

        ym = loess.n_yy[idxs]
        xmt_wm = Hermitian(xm'wm)
        beta = inv(xmt_wm * xm) * xmt_wm * ym
        y = (beta * xp)[1]
    else
        xx = loess.n_xx[idxs]
        yy = loess.n_yy[idxs]
        sum_weight    = sum( weights)
        sum_weight_x  = sum( xx .* weights)
        sum_weight_y  = sum( yy .* weights)
        sum_weight_x2 = sum((xx .* xx) .* weights)
        sum_weight_xy = sum((xx .* yy) .* weights)

        mean_x = sum_weight_x / sum_weight
        mean_y = sum_weight_y / sum_weight

        @show b = ((sum_weight_xy - mean_x * mean_y * sum_weight) /
            (sum_weight_x2 - mean_x * mean_x * sum_weight))
        @show a = mean_y - b * mean_x
        @show y = a + b * n_x
    end
        
    denormalize_y(loess, y)
end
```

```julia
xx = [0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
      4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
      8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
      4.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
      18.7572812]

yy = [18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
      213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
      227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
      160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
      243.18828]

loess = Loess(xx, yy)

x = 0.5578196
x, estimate(loess, x, window=7, use_matrix=false, degree=1)
```

```julia
#y = Float64[]
#for x in sort(xx)
#    push!(y,estimate(loess, x, window=7, use_matrix=true, degree=1))
#end

scatter( xx, yy)
```

```julia

```
