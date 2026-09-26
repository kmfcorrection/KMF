"""HILP -- Hessian-Induced Local Prior for deterministic PDE surrogates.

Post-hoc construction of a Gaussian predictive approximation from the local
Jacobian geometry of a frozen deterministic model, and a physics-constrained
posterior built on top of it. See docs/method.md.
"""
from . import (baselines, calibrate, curvature, data, metrics, models, physics,
               posterior, priors, samplers, toy, utils)

__all__ = ["baselines", "calibrate", "curvature", "data", "metrics", "models",
           "physics", "posterior", "priors", "samplers", "toy", "utils"]
__version__ = "0.1.0"
