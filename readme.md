# Stochastic Lanczos REML algorithms

## `SL_REML v0.1a`

Code accompanying _[Stochastic Lanczos estimation of genomic variance components for linear mixed-effects models](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2978-z)_.

 - `L_Seed` constructs bases for Krylov subspaces:  B, (A+σI)B, (A+σI)²B, ...

 - `L_Solve` solves (A + σI)X = B using results from `L_Seed`

 - `SLQ_LDet` returns approximate log det (A + σI) given spectal decompositions of Jacobi matrices from Lanczos decompositions of seed Krylov subspaces for probe vectors

 - `L_FOMC_REML` extension of BOLT-LMM algorithm to recycle Krylov subspace bases involved in solving linear systems

 - `SLDF_REML` zero order REML estimation using (shifted) block Lanczos conjugate gradients and (shifted) stochastic Lanczos quadrature

## `SL_REML v0.1b`
High performance variant of Stochastic First-order Derivative-Free REML algorithm (`SLDF_REML`) as employed in _[Assortative mating biases marker-based heritability estimators](https://doi.org/10.1101/2021.03.18.436091)_.

 - `L_Seed_numba` compiled variant of `L_Seed` above

 - `L_Solve_numba` compiled variant of `L_Seed` above

 - `SLQ_LDet` see above
 
 - `SLDF_REML_numba` compiled variant of `SLDF_REML` above
 
In contrast to `SLDF_REML`, `SLDF_REML_numba` assumes all covariates have been projected out of the genotype and phenotype data

