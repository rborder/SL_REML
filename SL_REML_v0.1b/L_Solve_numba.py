#!/usr/bin/python
from numpy import *
from numpy import linalg as nla
import numba as nb

######################## L_Solve #########################
## solves (A + sigma I)X = B using results from L_Seed) ##
##########################################################
@nb.njit(parallel=False)
def L_Solve(
        U: ndarray,
        beta: ndarray,
        delta_seed: ndarray,
        rho: ndarray,
        B: ndarray,        # RHS
        sigma: float,          # shift >=0
        tol = 5e-3,        # absolute tolerance for CG
        verbose= False,    # verbose output
        p_freq = 5         # print frequency (if verbose)
    ) -> ndarray:

    delta = delta_seed + sigma
    n = U.shape[0]
    t = U.shape[1]
    maxit = U.shape[2]

    X = zeros((n,t,maxit)) ## approximate soln
    P = zeros((n,t,maxit)) ## search directions
    R = zeros((n,t,maxit)) ## residual

    ## coefficients
    omega = zeros((t,maxit))
    gamma =  ones((t,maxit))

    ## initial values
    R[:,:,0] = B
    P[:,:,0] = B

    j = 0
    cnvg = False
    res_norms = zeros(t, dtype=float32)

    while j < maxit-1:

        gamma[:,j] = (delta[:,j] - omega[:,j-1]/gamma[:,j-1])**-1
        omega[:,j] = (beta[:,j+1]*gamma[:,j])**2
        rho[:,j+1] = -beta[:,j+1]*gamma[:,j]*rho[:,j]

        ## CG vectors update
        X[:,:,j+1] = X[:,:,j] + gamma[:,j]*P[:,:,j]
        R[:,:,j+1] = rho[:,j+1]*U[:,:,j+1]
        P[:,:,j+1] = R[:,:,j+1] + omega[:,j]*P[:,:,j]

        j += 1

        for i in nb.prange(t):
            res_norms[i] = nla.norm(R[:,i,j])
        res_norm = amax(res_norms)

        if j % p_freq ==0 : print("Error at step ",j,
                                  " is ", res_norm)
        if res_norm <= tol:
            cnvg = True
            break

    if verbose and cnvg: print("Converged after ",j," iterations.")
    elif verbose: print("Failed to converge after ",j," iterations.")

    return X[:,:,j]
