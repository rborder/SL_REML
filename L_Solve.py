#!/usr/bin/python
from numpy import *
from scipy import linalg as la
from scipy.sparse import linalg as spla

######################## L_Solve #########################
## solves (A + sigma I)X = B using results from L_Seed) ##
##########################################################
def L_Solve(
        seedSystem: dict,  # ouput from L_Seed
        B: ndarray,        # RHS
        σ: float,          # shift >=0
        tol = 5e-3,        # absolute tolerance for CG
        verbose= False,     # verbose output
        p_freq = 5        # print frequency (if verbose)
    ) -> ndarray:

    U = seedSystem["U"]
    δ = seedSystem["δ"] + σ
    β = seedSystem["β"]
    ρ = seedSystem["ρ"]
    n = U.shape[0]
    t = U.shape[1]
    maxit = U.shape[2]

    X = zeros((n,t,maxit)) ## approximate soln
    P = zeros((n,t,maxit)) ## search directions
    R = zeros((n,t,maxit)) ## residual

    ## coefficients
    ω = zeros((t,maxit))
    γ =  ones((t,maxit))

    ## initial values
    R[:,:,0] = B
    P[:,:,0] = B

    j = 0
    cnvg = False

    while j < maxit-1:

        γ[:,j] = (δ[:,j] - ω[:,j-1]/γ[:,j-1])**-1
        ω[:,j] = (β[:,j+1]*γ[:,j])**2
        ρ[:,j+1] = -β[:,j+1]*γ[:,j]*ρ[:,j]

        ## CG vectors update
        X[:,:,j+1] = X[:,:,j] + γ[:,j]*P[:,:,j]
        R[:,:,j+1] = ρ[:,j+1]*U[:,:,j+1]
        P[:,:,j+1] = R[:,:,j+1] + ω[:,j]*P[:,:,j]

        j += 1

        # res_norm = amax(abs(R[:,:,j]))
        res_norm = max(apply_along_axis(la.norm,0, R[:,:,j]))

        if verbose and j % p_freq ==0 : print("Error at step ",j,
                                  " is ", res_norm)
        if res_norm <= tol:
            cnvg = True
            break

    if verbose and cnvg: print("Converged after ",j," iterations.")
    elif verbose: print("Failed to converge after ",j," iterations.")

    return X[:,:,j]
