#!/usr/bin/python
from numpy import *
from scipy import linalg as la
from scipy import optimize as opt
from scipy.sparse import linalg as spla
from L_Seed import L_Seed
from L_Solve import L_Solve
import pytictoc

############################ L_FOMC_REML #################################
## extension of BOLT-LMM algorithm to recycle Krylov subspace bases     ##
## involved in solving linear systems                                   ##
##########################################################################
def L_FOMC_REML(
    Z: ndarray,      # standardized genotype matrix / sqrt(m)
    X: ndarray,      # covariate matrix
    y: ndarray,      # phenotype vector
    nMC = 15,        # number of MC samples
    s2max = .7,      # maximal heritable VC value
    ZZ = False,      # optional precomputed relatedness matrix
    maxIter = 15,  # max secant iterations
    tol_L = 5e-3,    # abs. lanczos tolerance
    tol_VC = 1e-5,   # abs. var. component tolerance
    verbose = True,  # verbose output
    p_freq = 5,      # print frequency
    timing = False,  # return timing information?
    seed = None      # seed for MC sample
    ):

    ## initialize timer
    if timing: TT = pytictoc.TicToc(); TT.tic()

    n,c = X.shape
    m = Z.shape[1]

    ## qr decomposition of covariate matrix
    Q = la.qr(X, mode = "economic")[0]

    ## project covariates out of phenotype vector
    def scale(x):
        return (x-mean(x))/std(x)
    y_proj = scale(y - Q @ (Q.T @ y))

    ## construct implicit linear operators
    def Z_proj_rmv(v): return (Z.T@(v - Q@(Q.T @ v)))/sqrt(m)
    def Z_proj_mv(v): tmp = Z@v; return (tmp - Q@(Q.T @ tmp))/sqrt(m)

    Z_proj = spla.LinearOperator(Z.shape, matvec = Z_proj_mv,
                                 matmat = Z_proj_mv,rmatvec = Z_proj_rmv)
    Z_proj_adj = spla.LinearOperator((Z.shape[1],Z.shape[0]), matvec = Z_proj_rmv,
                                     matmat = Z_proj_rmv, rmatvec = Z_proj_mv)

    ## use precomputed GRM if available
    if not isinstance(ZZ, bool):
        grmPrecomputed = True
        def ZZ_proj_mv(v): tmp = ZZ@(v - Q@(Q.T @ v)); return tmp - Q@(Q.T @ tmp)
        ZZ_proj = spla.LinearOperator(ZZ.shape, matvec = ZZ_proj_mv, matmat = ZZ_proj_mv)
    else:
        grmPrecomputed = False
        def ZZ_proj_mv(v): tmp = Z@(Z.T@(v - Q@(Q.T @ v))); return (tmp - Q@(Q.T @ tmp))/m
        ZZ_proj = spla.LinearOperator((n,n), matvec = ZZ_proj_mv, matmat = ZZ_proj_mv)


    f = zeros(maxIter)
    logTau = zeros(maxIter)
    n = Z_proj.shape[0]
    m = Z_proj.shape[1]

    ## random latent variables:
    random.seed(seed)
    u_MC = random.randn(m, nMC)
    e_MC = random.randn(n, nMC)
    e_MC = e_MC -Q@(Q.T@e_MC)
    ## precompute matvec
    Zu_MC = Z_proj @ u_MC

    ## solve seed systems
    τ0 = (1-s2max)/s2max
    H0 = spla.LinearOperator(ZZ_proj.shape,
                             matmat = lambda V : ZZ_proj@V + τ0*V,
                             matvec = lambda v : ZZ_proj@v + τ0*v)
    seed_y = L_Seed(H0, y_proj, tol = tol_L, verbose = verbose,
                    p_freq = p_freq)
    seed_e_MC = L_Seed(H0, e_MC, tol = tol_L, verbose = verbose,
                    p_freq = p_freq)
    seed_Zu_MC = L_Seed(H0, Zu_MC, tol = tol_L, verbose = verbose,
                    p_freq = p_freq)

    ## construct objective function:
    def f_reml(s2g):
        τ = (1-s2g)/s2g
        σ = τ - τ0

        ## recycle bases to resolve shifted systems:
        soln = L_Solve(seed_y, y_proj, σ, tol = tol_L,
                       verbose = verbose, p_freq = p_freq)
        soln_Zu_MC = L_Solve(seed_Zu_MC, Zu_MC, σ, tol = tol_L,
                             verbose = verbose, p_freq = p_freq)
        soln_e_MC = L_Solve(seed_e_MC, e_MC, σ, tol = tol_L,
                            verbose = verbose, p_freq = p_freq)


        ## compute BLUPs
        u_BLUP = Z_proj_adj @ (soln)
        e_BLUP = sqrt(τ)*soln

        HinvY_MC = soln_Zu_MC + sqrt(τ)*soln_e_MC
        u_MC_BLUP = Z_proj_adj @ (HinvY_MC)
        e_MC_BLUP = sqrt(τ)*HinvY_MC

        ## criterion for root finding:
        return log(sum(u_BLUP**2)*sum(e_MC_BLUP**2)/(
            sum(e_BLUP**2)*sum(u_MC_BLUP**2)))

    ## initial overhead timing
    if timing:
        overhead = TT.tocvalue()
        output, rr = opt.brentq(f_reml,0.01,s2max,xtol=tol_VC,disp=verbose, full_output = True)
        J = rr.iterations
    else:
        output = opt.brentq(f_reml,0.01,s2max,xtol=tol_VC,disp=verbose)
        J = 0


    if timing: ## return detailed output if timing is enabled
        mainloop = TT.tocvalue() - overhead  ## subsequent iteration timing
        if not isinstance(seed, int): seed = -1
        return {'soln':output,
                'method':"L_MCNR_REML",
                'nIteration':J,
                'overhead':overhead,
                'mainloop':mainloop,
                'n':n,
                'm':m,
                'c':c,
                'nRand':nMC,
                'grmPrecomputed':grmPrecomputed,
                'tol_L':tol_L,
                'tol_VC':tol_VC,
                'seed':seed}
    else: ## else return VC estimate
        return output
