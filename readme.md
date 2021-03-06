# Stochastic Lanczos REML algorithms

Code accompanying _[Stochastic Lanczos estimation of genomic variance components for linear mixed-effects models](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2978-z)_.


 - `L_Seed` constructs bases for Krylov subspaces:  B, (A+σI)B, (A+σI)²B, ...

 - `L_Solve` solves (A + σI)X = B using results from `L_Seed`

 - `SLQ_LDet` returns approximate log det (A + σI) given spectal decompositions of Jacobi matrices from Lanczos decompositions of seed Krylov subspaces for probe vectors

 - `L_FOMC_REML` extension of BOLT-LMM algorithm to recycle Krylov subspace bases involved in solving linear systems

 - `SLDF_REML` zero order REML estimation using (shifted) block Lanczos conjugate gradients and (shifted) stochastic Lanczos quadrature
