"""Superseded in the October 2026 rerun: the agg stage of mc.py now computes rr["gbc_nomicro"] itself (same method).
GBC without micro moments in the Monte Carlo, for a like for like comparison with BLP GMM,
which uses aggregate data only. Appends to results/mc.pkl."""
import pickle, numpy as np
from estimators import GBCTargets, PARAMS
from mc import TRUE

R = pickle.load(open("results/mc.pkl", "rb"))
ia = [PARAMS.index(k) for k in ("alpha", "sp", "sx")] + [12, 13, 14]
for T in [40, 15]:
    gt = GBCTargets(R[T]["S"][:, :-2], R[T]["Y"])
    for rr in R[T]["reps"]:
        rr["gbc_nomicro"] = gt.posterior(rr["S"][:-2])[:, ia]
with open("results/mc.pkl", "wb") as f:
    pickle.dump(R, f)
