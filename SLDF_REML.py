#!/usr/bin/python
from numpy import *
from numpy import linalg as npla
from scipy import linalg as la
from scipy import optimize as opt
from scipy.sparse import linalg as spla
from L_Seed import L_Seed
from SLQ_LDet import SLQ_LDet
from L_Solve import L_Solve
import pytictoc


############################# SLDF_REML ##############################
## zero order REML estimation using (shifted) Lanczos conjugate     ##
## gradients and (shifted) stochastic Lanczos quadrature            ##
######################################################################
def SLDF_REML(
    Z,               # standardized genotype matrix
    X: ndarray,      # covariate matrix
    y: ndarray,      # phenotype vector
    m: int,          # number of markers
    ZZ = False,      # (optional) precomputed GRM
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
    def scale(x):
        return (x-mean(x))/std(x)
    ## extract needed dimensions
    n,c = X.shape

    ## construct implicit GRM if needed,
    ## ensuring division by m occurs after matvecs
    if isinstance(ZZ, bool):
        grmPrecomputed = False
        def ZZ_mv(v): return (Z @ (Z.T @ v))/m
        ZZ_proj = spla.LinearOperator((n,n), matmat = ZZ_mv, matvec = ZZ_mv)
    else:
        def ZZ_mv(v): return ZZ @ v
        ZZ_proj = spla.LinearOperator((n,n), matmat = ZZ_mv, matvec = ZZ_mv)
        grmPrecomputed = True

    ## qr decomposition of covariate matrix
    Q = la.qr(X, mode = "economic")[0]
    ## apply projection to phenotype vector
    y_proj = y - Q @ (Q.T @ y)

    ## sample normalized Rademacher probing vetors
    random.seed(seed)
    V = array([(random.binomial(1,.5,size=n)*2 - 1) for k in arange(0,n_V)]).T
    V = apply_along_axis(lambda x: x/la.norm(x),0,V)


    ## represent seed system LHSs as implicit linear operators
    τ0 = (1-s2max)/s2max
    def H0_quadform_mv(v):
        ZPv = ZZ_proj@(v - Q @ (Q.T @ v))
        return ZPv - Q @ (Q.T @ ZPv) + τ0*v
    H0_quadform = spla.LinearOperator((n,n),
                                      matmat = H0_quadform_mv,
                                      matvec = H0_quadform_mv)
    def H0_ldet_mv(v): return ZZ_proj@v + τ0*v
    H0_ldet = spla.LinearOperator((n,n),
                                  matmat = H0_ldet_mv,
                                  matvec = H0_ldet_mv)


    ## Lanczos decompositions of seed systems
    seed_y = L_Seed(H0_quadform, y_proj, tol = tol_L,
                    verbose = verbose, p_freq = p_freq)
    seed_X = L_Seed(H0_ldet, X, tol = tol_L,
                    verbose = verbose, p_freq = p_freq)
    seed_V = L_Seed(H0_ldet, V, tol = tol_L,
                    verbose = verbose, p_freq = p_freq)

    ## decompose Jacobi matrices for SLQ
    W_V = zeros([n_V,seed_V['δ'].shape[1]])  ## eigenvalues
    D_V = zeros([n_V,seed_V['δ'].shape[1]])  ## first elements of eigenvectors
    for l in arange(0,n_V):
        D_V[l,:], tmpW  = la.eigh_tridiagonal(seed_V["δ"][l,:], seed_V["β"][l,1:])
        W_V[l,:]        = tmpW[0,:]

    def REML_criterion(ss):
        global s2
        τ = (1-ss)/ss
        σ = τ - τ0
        yPy = (τ*y_proj.T @ L_Solve(seed_y, y_proj, σ))[0][0]

        v_e = yPy/(n-c)
        v_g = v_e/τ
        s2 = v_g/(v_g+v_e)

        ldet = npla.slogdet(X.T @ L_Solve(seed_X, X, σ))[1] + SLQ_LDet(D_V, W_V, n, n_V, σ)+(n-c)*log(v_e)-(n-c)*log(τ)

        print("heritability estimate: ", s2)
        return ldet + yPy/v_e


    ## initial overhead timing
    if timing: overhead = TT.tocvalue()

    output = opt.fminbound(REML_criterion,s2min,s2max,
                           xtol = tol_VC, maxfun = maxIter,
                           disp = 3*verbose, full_output = timing)


    if timing: ## return detailed output if timing is enabled
        mainloop = TT.tocvalue() - overhead  ## subsequent iteration timing
        if not isinstance(seed, int): seed = -1
        return {'soln':s2,
                'method':"L_DF_REML_1",
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
