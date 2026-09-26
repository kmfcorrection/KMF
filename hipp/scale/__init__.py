"""Scale-out layer: everything needed to run HILP on a real PDE foundation model.

The 1D code in `hipp/` forms dense N x N Jacobians, Cholesky factors and
eigendecompositions. That is deliberate -- at N = 128 the exact object is
affordable and every approximate estimator can be checked against ground truth.
It is also the reason none of it runs at foundation-model scale: a 2D state with
C channels on an H x W grid has N = C*H*W (49,152 for 3 x 128 x 128), so a dense
N x N covariance is 2.4e9 entries and its Cholesky is ~1e14 flops.

This package re-expresses the same method in a representation that never
materializes an N x N matrix:

    Sigma = alpha * (U diag(d) U^T + tau I),   U in R^{N x k} orthonormal, k << N

Both curvature surrogates map onto it (see `lowrank.py`), every quantity the
method needs is O(N k), and the only dense linear algebra left is k x k.

Modules
-------
lowrank          the representation above and all Gaussian quantities on it
jacobian         batched matrix-free JVP/VJP + blocked randomized SVD
data2d           GPU pseudo-spectral 2D Navier-Stokes / Kolmogorov flow, PDEBench
physics2d        differentiable 2D residual energy
models2d         FNO2d / UNet2d surrogates that scale
adapters         uniform frozen-FM interface (local ckpts + pretrained FMs)
calibrate_scale  streaming one-parameter scale calibration
metrics_scale    NLL / coverage / CRPS / ECE / subspace-split chi^2 at scale
baselines_scale  isotropic, diagonal, low-rank fixed-global, ensemble, conformal
rollout          low-rank Lyapunov covariance propagation over autoregressive rollout
"""
