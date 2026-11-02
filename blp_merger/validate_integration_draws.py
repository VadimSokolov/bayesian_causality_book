"""Draw sets of validate_integration.py (200, 800, 3,200 antithetic draws), importable without running the check."""
import numpy as np
from blp_core import Design

D0 = Design(seed=0)


def anti(nu, eta):
    return (np.concatenate([nu, -nu, nu, -nu]), np.concatenate([eta, eta, -eta, -eta]))


rng99 = np.random.default_rng(99)
base800 = (np.concatenate([D0.nu, rng99.normal(size=600)]), np.concatenate([D0.eta, rng99.normal(size=600)]))
SETS = {200: (D0.nu, D0.eta), 800: anti(D0.nu, D0.eta), 3200: anti(*base800)}


def design(n):
    d = Design(seed=0)
    d.nu, d.eta = SETS[n]; d.ns = len(d.nu)
    return d


