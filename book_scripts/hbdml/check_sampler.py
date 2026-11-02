#!/usr/bin/env python3
"""Check that the Gaussian draw used by the horseshoe samplers has covariance inv(A).

Draws 200,000 times from N(m, A^{-1}) for a fixed 2 x 2 precision A with the upper and the lower Cholesky
factor (draw_from_precision in sim_cross_section.py) and asserts that the sample covariance matches inv(A)
to 1 percent (relative to the largest entry of inv(A)) and the sample mean matches A^{-1} b. It also reports
the covariance of the earlier, wrong draw U^{-T} z, which is (U U^T)^{-1}, to show that the check can fail.

Usage: timeout 600 python3 -I book_scripts/hbdml/check_sampler.py   (run from the repository root or from this folder)
"""
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_cross_section import draw_from_precision

A = np.array([[4.0, 1.5], [1.5, 1.0]])
b = np.array([1.0, -2.0])
target = np.linalg.inv(A)
mean = target @ b
N = 200_000
np.random.seed(1)

ok = True
for lower in (False, True):
    X = np.array([draw_from_precision(A, b, lower=lower) for _ in range(N)])
    C = np.cov(X.T)
    rel = np.max(np.abs(C - target)) / np.max(np.abs(target))
    mrel = np.max(np.abs(X.mean(0) - mean)) / np.max(np.abs(mean))
    print("lower=%s: max relative covariance error %.4f, mean error %.4f" % (lower, rel, mrel))
    ok &= bool(rel < 0.01) and bool(mrel < 0.01)

U, low = cho_factor(A)
Z = np.random.randn(N, 2)
Xbad = (solve_triangular(U, Z.T, lower=low, trans='T')).T
Cbad = np.cov(Xbad.T)
relbad = np.max(np.abs(Cbad - target)) / np.max(np.abs(target))
print("earlier draw U^{-T} z: max relative covariance error %.4f (this is the bug)" % relbad)
print("inv(A) =", target.round(4).tolist())
assert ok, "draw_from_precision does not reproduce inv(A) to 1 percent"
assert relbad > 0.05
print("PASS")
