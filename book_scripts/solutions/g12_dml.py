"""ex:ch04:5: the aggregation rule of Chapter dml on the ten splits of g1_check.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "book_scripts"))
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import ric_sim
rng = np.random.default_rng(ric_sim.SEED0 + 1); n, p = 200, 50
X = rng.normal(size=(n, p)); g = np.zeros(p); g[:5] = 1
t = X @ g + rng.normal(size=n); eps = rng.normal(size=n); y = t + X @ g + eps
out = []
for s in range(1, 11):
    ry = np.zeros(n); rt = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
        ry[te] = y[te] - LassoCV(cv=5).fit(X[tr], y[tr]).predict(X[te])
        rt[te] = t[te] - LassoCV(cv=5).fit(X[tr], t[tr]).predict(X[te])
    th = rt @ ry / (rt @ rt); psi = rt * (ry - th * rt); out.append((th, np.sqrt((psi ** 2).sum()) / (rt @ rt)))
a = np.array(out); med = np.median(a[:, 0])
print("est", np.round(np.sort(a[:, 0]), 3), "se range", a[:, 1].min().round(3), a[:, 1].max().round(3), "mean se", a[:, 1].mean().round(4), "median se", np.median(a[:, 1]).round(4))
print("median", med.round(4), "sd", a[:, 0].std(ddof=1).round(4), "range", np.ptp(a[:, 0]).round(4))
print("agg se", np.sqrt(np.median(a[:, 1] ** 2 + (a[:, 0] - med) ** 2)).round(5))
