#!/usr/bin/python
from numpy import *
from numpy import linalg as npla
from scipy import linalg as la
from scipy import optimize as opt
from scipy.sparse import linalg as spla
from L_Seed_numba import L_Seed
from SLQ_LDet import SLQ_LDet
from L_Solve_numba import L_Solve
import pytictoc
import numba as nb
# nb.config.THREADING_LAYER = 'safe'


############################# SLDF_REML ##############################
## zero order REML estimation using (shifted) Lanczos conjugate     ##
## gradients and (shifted) stochastic Lanczos quadrature            ##
######################################################################
    # n = A.shape[0]
    # t = B.shape[1]
    # X = zeros((n,t,maxit), dtype=float32) # CG approximate solutions
    # R = zeros((n,t,maxit), dtype=float32) # CG residuals
    # U = zeros((n,t,maxit), dtype=float32) # orthonormal bases
    # ## coefficients
    # rho = zeros((t,maxit), dtype=float32)
    # beta = zeros((t,maxit), dtype=float32)
    # omega = zeros((t,maxit), dtype=float32)
    # gamma =  ones((t,maxit), dtype=float32)
    # delta = zeros((t,maxit), dtype=float32)

def scale(x):
    return (x - mean(x))/std(x)

    # s2max = .7;      # maximal heritable VC value
    # s2min = .1;      # minimal heritable VC value
    # n_V = 15;        # number of random probes
    # tol_L = 1e-9;    # abs. lanczos tolerance
    # tol_VC = 1e-5;   # abs. var. component tolerance
    # maxIter = 15;    # max opt iterations
    # verbose = True;  # verbose output
    # p_freq = 5;      # print frequency
    # timing = False;  # return timing information?
    # seed = None      # seed for MC sample
 
def SLDF_REML(
    ZZ,              # adjusted standardized genotype matrix
    y: ndarray,      # phenotype vector
    m: int,          # number of markers
    s2max = .7,      # maximal heritable VC value
    s2min = .1,      # minimal heritable VC value
    n_V = 15,        # number of random probes
    tol_L = 1e-9,    # abs. lanczos tolerance
    tol_VC = 1e-5,   # abs. var. component tolerance
    maxIter = 15,    # max opt iterations
    verbose = True,  # verbose output
    p_freq = 5,      # print frequency
    timing = False,  # return timing information?
    seed = None      # seed for MC sample
    ):

    ## initialize timer
    if timing: TT = pytictoc.TicToc(); TT.tic()

    ## extract needed dimensions
    n = ZZ.shape[0]

    ## qr decomposition of covariate matrix (intercept only in this case)
    X=ones((n,1),dtype=float32)
    Q = la.qr(X, mode = "economic")[0]

    ## apply projection to phenotype vector
    y_proj = y - Q @ (Q.T @ y)

    ## sample normalized Rademacher probing vetors
    random.seed(seed)
    V = array([(random.binomial(1,.5,size=n)*2 - 1) for k in arange(0,n_V)]).astype(float32).T
    V = apply_along_axis(lambda x: (x/la.norm(x)),0,V).astype(float32)

    ## represent seed system LHSs as implicit linear operators
    tau0 = float32((1-s2max)/s2max)

    ## Lanczos decompositions of seed systems
    y_U,y_beta,y_delta,y_rho = L_Seed(ZZ, y_proj, Q, tau0, tol =  float32(tol_L),
                                      p_freq = p_freq, qform=True)
    print('seed_y')
    X_U,X_beta,X_delta,X_rho  = L_Seed(ZZ, X, Q, tau0, tol =  float32(tol_L),
                    p_freq = p_freq, qform=False)
    print('seed_X')
    V_U,V_beta,V_delta,V_rho  = L_Seed(ZZ, V, Q, tau0, tol =  float32(tol_L),
                    p_freq = p_freq, qform=False)
    print('seed_V')

    ## decompose Jacobi matrices for SLQ
    W_V = zeros([n_V,V_delta.shape[1]],dtype=float32)  ## eigenvalues
    D_V = zeros([n_V,V_delta.shape[1]],dtype=float32)  ## first elements of eigenvectors
    for l in arange(0,n_V):
        D_V[l,:], tmpW  = la.eigh_tridiagonal(V_delta[l,:], V_beta[l,1:])
        W_V[l,:]        = tmpW[0,:].astype(float32)
    c=1
    def REML_criterion(ss):
        global s2
        tau = (1-ss)/ss
        sigma = tau - tau0
        yPy = (tau*y_proj.T @ L_Solve(y_U,y_beta,y_delta,y_rho, y_proj, sigma))[0][0]
        print(yPy)

        v_e = yPy/(n-c)
        v_g = v_e/tau
        s2 = v_g/(v_g+v_e)

        ldet = npla.slogdet(X.T @ L_Solve(X_U,X_beta,X_delta,X_rho, X, sigma))[1] + SLQ_LDet(D_V, W_V, n, n_V, sigma)+(n-c)*log(v_e)-(n-c)*log(tau)

        if verbose: print("heritability estimate: ", s2)
        return ldet + yPy/v_e

    ## initial overhead timing
    if timing: overhead = TT.tocvalue()

    ## optimize REML criterion
    output = opt.fminbound(REML_criterion,s2min,s2max,
                           xtol = tol_VC, maxfun = maxIter,
                           disp = 3*verbose, full_output = timing)


    if timing: ## return detailed output if timing is enabled
        mainloop = TT.tocvalue() - overhead  ## subsequent iteration timing
        if not isinstance(seed, int): seed = -1
        return {'soln':s2,
                'nIteration':output[3],
                'overhead':overhead,
                'mainloop':mainloop,
                'n':n,
                'm':m,
                'c':c,
                'nRand':n_V,
                'grmPrecomputed':grmPrecomputed,
                'tol_L':tol_L,
                'tol_VC':tol_VC,
                'seed':seed}
    else: ## else return VC estimate
        return s2
