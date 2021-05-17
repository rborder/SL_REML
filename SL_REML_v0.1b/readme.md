
## Toy example demonstrating application of `SLDF_REML_numba` 


```python3
import numpy as np
from scipy import linalg as la
from SLDF_REML_numba import SLDF_REML

# set random state
np.random.seed(123)

n = 1000 # number of individuals
m = 100  # number of variants
h2 = .4  # heritability
c = 10   # number of covariates

## fixed effects
X = np.random.randn(n,c)
b = np.linspace(.1,1,c).reshape(c,1)

## genotypes
Z = np.random.randn(n,m)
## genetic effects
u = np.random.randn(m) * np.sqrt(h2/m)

## residuals
e = np.random.randn(n) * np.sqrt(1-h2)

## phenotype
y = (X@b).reshape((n,1)) + (Z@u).reshape((n,1)) + e.reshape((n,1))

## format inputs
y = (y-np.mean(y))/np.std(y)
Y = y.reshape([y.shape[0],1]).astype(np.float32)
Z = (Z/np.sqrt(m)).astype(np.float32)

## project out covariates
Q = la.qr(X, mode = "economic")[0].astype(np.float32)
Y = Y - Q @ (Q.T @ Y)
Z = Z - Q @ (Q.T @ Z)
ZZ = Z @ Z.T

h2est = SLDF_REML(ZZ,Y,1, tol_L=np.float32(3e-4),tol_VC = np.float32(3e-4))
```

## Dependencies

Installation consists of installing dependences only.  Assuming python3 is present, all required dependencies can be installed in a few minutes using `pip` or `conda`:

 - `numba` v0.49.1+
 - `numpy` v1.18.4+
 - `scipy` v1.4.1+
 - `tbb` v2020.0.133+
 - `pytictoc` v1.4.0+

This software has been confirmed to work on
 - Ubuntu 18.04.5 LTS x86_64 with kernel 5.4.0-72-generic
 - Red Hat Enterprise Linux v7.6 x86_64 with kernel: 3.10.0-957.21.3.el7
