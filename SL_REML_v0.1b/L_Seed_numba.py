#!/usr/bin/python
from numpy import *
from numpy import linalg as nla
from scipy.sparse import linalg as spla
import numba as nb

nb.config.THREADING_LAYER = 'safe'

######################## L_Seed ##########################
## constructs bases for Krylov subspaces:               ##
## B, (A+sI)B, (A+sI)²B, ...                            ##
##########################################################
@nb.njit(parallel=False)
def L_Seed(
        A: ndarray,             # LHS
        B: ndarray,             # RHS
        Q,
        tau0,
        tol = float32(5e-4),    # absolute tolerance
        p_freq = 5,             # print frequency
        maxit = 100,             # maximum iteration count
        qform=True
    ):

    n = A.shape[0]
    t = B.shape[1]
    X = zeros((n,t,maxit), dtype=float32) # CG approximate solutions
    R = zeros((n,t,maxit), dtype=float32) # CG residuals
    U = zeros((n,t,maxit), dtype=float32) # orthonormal bases
    ## coefficients
    rho = zeros((t,maxit), dtype=float32)
    beta = zeros((t,maxit), dtype=float32)
    omega = zeros((t,maxit), dtype=float32)
    gamma =  ones((t,maxit), dtype=float32)
    delta = zeros((t,maxit), dtype=float32)
    ## initial values
    for i in nb.prange(t):
        rho[i,0] = nla.norm(B[:,i])
    beta[:,0] = rho[:,0]
    R[:,:,0] = U[:,:,0] = B
    U[:,:,0] = R[:,:,0]/beta[:,0]

    j = 0
    cnvg = 1

    while j < maxit:

        ## Lanczos iteration
        if qform:
            tmp = (A@(U[:,:,j] - Q @ (Q.T @ U[:,:,j]))) - Q @ (Q.T @ (A@(U[:,:,j] - Q @ (Q.T @ U[:,:,j])))) + tau0*U[:,:,j]
        else:
            tmp = A @ U[:,:,j] +  tau0*U[:,:,j] ## slow [part]

        delta[:,j] = diag(U[:,:,j].T @ tmp)
        tmp = tmp - U[:,:,j]*delta[:,j] - beta[:,j]*U[:,:,j-1]
        for i in nb.prange(t):
            beta[i,j+1] = nla.norm(tmp[:,i])
        U[:,:,j+1] = tmp / beta[:,j+1]

        ## CG coefficents update
        gamma[:,j] = (delta[:,j] - omega[:,j-1]/gamma[:,j-1])**-1
        omega[:,j] = (beta[:,j+1]*gamma[:,j])**2
        rho[:,j+1] = -beta[:,j+1]*gamma[:,j]*rho[:,j]

        ## CG vectors update
        R[:,:,j+1] = rho[:,j+1]*U[:,:,j+1]

        j += 1

        res_norms = zeros(t, dtype=float32)
        for i in nb.prange(t):
            res_norms[i] = nla.norm(R[:,i,j])
        res_norm = amax(res_norms)

        if j % p_freq == 0 : print("Error at step ", j,
                                  " is ", res_norm)
        if res_norm < tol:
            cnvg = 0
            break

    if cnvg==0: print("Converged after ",j," iterations.")
    else: print("Failed to converge after ",j," iterations.")

    return U[:,:,0:j], beta[:,0:j], delta[:,0:j], rho[:,0:j]
