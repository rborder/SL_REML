#!/usr/bin/python
from numpy import *
from scipy import linalg as la
from scipy.sparse import linalg as spla

######################## L_Seed ##########################
## constructs bases for Krylov subspaces:               ##
## B, (A+sI)B, (A+sI)²B, ...                            ##
##########################################################
def L_Seed(
        A: spla.LinearOperator, # LHS
        B: ndarray,             # RHS
        tol = 5e-3,             # absolute tolerance
        p_freq = 5,             # print frequency
        maxit = 50,             # maximum iteration count
        verbose = True          # print extra information
    ) -> dict:
    """ see alg * in the paper """


    n = A.shape[0]
    t = B.shape[1]

    X = zeros((n,t,maxit)) # CG approximate solutions
    R = zeros((n,t,maxit)) # CG residuals
    U = zeros((n,t,maxit)) # orthonormal bases

    ## coefficients
    ρ = zeros((t,maxit))
    β = zeros((t,maxit))
    ω = zeros((t,maxit))
    γ =  ones((t,maxit))
    δ = zeros((t,maxit))

    ## initial values
    ρ[:,0] = apply_along_axis(la.norm,0,B)
    β[:,0] = ρ[:,0]
    R[:,:,0] = U[:,:,0] = B
    U[:,:,0] = R[:,:,0]/β[:,0]

    j = 0
    cnvg = False

    while j < maxit-1:

        ## Lanczos iteration
        tmp = A @ U[:,:,j]
        δ[:,j] = diag(U[:,:,j].T @ tmp)
        tmp = tmp - U[:,:,j]*δ[:,j] - β[:,j]*U[:,:,j-1]
        β[:,j+1] = apply_along_axis(la.norm,0,tmp)
        U[:,:,j+1] = tmp / β[:,j+1]

        ## CG coefficents update
        γ[:,j] = (δ[:,j] - ω[:,j-1]/γ[:,j-1])**-1
        ω[:,j] = (β[:,j+1]*γ[:,j])**2
        ρ[:,j+1] = -β[:,j+1]*γ[:,j]*ρ[:,j]

        ## CG vectors update
        R[:,:,j+1] = ρ[:,j+1]*U[:,:,j+1]

        j += 1

        res_norm = amax(apply_along_axis(la.norm,0,(R[:,:,j])))

        if verbose and j % p_freq ==0 : print("Error at step ",j,
                                  " is ", res_norm)
        if res_norm <= tol:
            cnvg = True
            break

    if verbose and cnvg: print("Converged after ",j," iterations.")
    elif verbose: print("Failed to converge after ",j," iterations.")

    return {
        "U":U[:,:,0:j],     ## orthonormal bases for K-subspaces
        "β":β[:,0:j],       ## Jacobi coefficients for Lanczos polys
        "δ":δ[:,0:j],
        "ρ":ρ[:,0:j]
        }
