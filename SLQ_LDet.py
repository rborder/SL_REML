#!/usr/bin/python
from numpy import *
from scipy import linalg as la
from scipy.sparse import linalg as spla



######################## SLQ_LDET #################################
## returns approximate log det (A + σI) given spectral           ##
## decompositions of Jacobi matrices from Lanczos decompositions ##
## of seed Krylov subspaces for probes                           ##
###################################################################
def SLQ_LDet(
        D: ndarray,     ## eigenvalues of Jacobi matrices
        W: ndarray,     ## eigenvectors of Jacobi matrices
        n: int,         ## dimension of A
        n_V: int,       ## number of probing vectors
        σ = 0           ## shift
        ) -> float:

    soln = 0                   ## storage for approximate trace

    for l in arange(0, n_V):   ## perform SLQ
        soln += (W[l,:]**2)@log(D[l,:] + σ)

    trEst = n/n_V * soln       ## estimate of Tr logm (A+ σI)

    return trEst
