"""
    RandomLM(;ny=100, p=200, sy=.1, nbet=5, mbet=2, sdbet=0, Xbr=0.05)

Randomly generates a linear model.

| Regressors  |   Coefficients      |
|-------------|---------------------|
| X[0:nbet]   |   Normal(mbet,sbet) |
| X[nbet:p]   |   0    nbet         |

If dupl=True, the second half of regressors has collinearity with the first half:

```
X[p/2:p]= X[0:p/2]+ Normal(0,Xbr)
```

## Parameters
  - ny: Number of observations.
  - p: Number of variables.
  - nbet: Number of non-zero coefficients
  - mbet: Mean value of these
  - sdbet: Std of these
  - Xbr: Std for collinearity
  - sy: Variance of u

## Returns
  - X : array, shape(n_samples,n_features)
    Regressor matrix. iid N(0,1) but columns are centered.
    If Xbr>0 each entry in the second half of the matrix
    is the one in the first half plus a noise of amplitude Xbr:
```
X[j,k+p/2]= X[j,k]+ Normal(0,Xbr)         k<p/2.
```
  - betas : array, shape(n_features)
    Coefficients. iid N(mbet,sbet). betas[i]=0 for i>=nbet.
  - y : array, shape(n_samples)
    Responses
"""
function RandomLM(; ny = 100, p = 200, sy = 0.1, nbet = 5, mbet = 2, sdbet = 0, Xbr = 0.05)

    betas = randn(nbet) .* sdbet .+ mbet
    length = nbet÷2:nbet
    betas[length] = -betas[length]
    betas = vcat(betas, zeros(Float64, p - nbet))
    X = randn(ny, p)
    if Xbr > 0
        p0 = p ÷ 2
        X[:, p-p0:p] .= X[:, 0:p0] + Xbr .* randn(ny * p0)
    end
    X .= X .- mean(X, dims = 1)
    y = X * betas + randn(ny) .* sy
    return X, y .- mean(y), betas

end
