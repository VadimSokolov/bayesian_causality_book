"""ex:chapter_blp:9. kappa ~ U(0,1) with the prior of estimators.py; same simulator, summaries, GBCTargets. Bertrand minivan data of run_experiments.py.
Usage from the repository root: python book_scripts/solutions/blp_kappa.py chunk C   (simulates 200 markets with indices 200 C to 200 C + 199 into results/solutions/ka/C.pkl; the full run uses C = 0 to 149, 30,000 draws)
then python book_scripts/solutions/blp_fit.py."""
import os, sys, json, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
sys.dont_write_bytecode = True; sys.path.insert(0, os.path.join(ROOT, "blp_merger"))
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_CONCURRENT_THREADS"): os.environ.setdefault(k, "1")
import numpy as np
from blp_core import Design, simulate_market
from estimators import draw_prior, summaries, PARAMS, GBCTargets
class D2(Design):
    kappa = None
    def ownership(self, conduct, firms=None):
        if conduct == "kappa":
            f = self.firms if firms is None else np.asarray(firms); own = (f[:, None] == f[None, :]).astype(float)
            return own + self.kappa * (1 - own)
        return super().ownership(conduct, firms)
D = D2(seed=0)
NC = 150
def one(i):
    rng = np.random.default_rng(910_000 + i)
    for _ in range(20):
        th = draw_prior(rng); D.kappa = rng.uniform(0, 1)
        r = simulate_market(D, th, "kappa", rng)
        if r["ok"]:
            S = summaries(D, r["s"], r["p"], r["micro"] + rng.normal(0, [0.03, 0.10]))
            if np.all(np.isfinite(S)): return [th[k] for k in PARAMS] + [D.kappa], S
    return None
TRUE = dict(b0=2.0, b1=1.0, b2=0.5, alpha=1.0, sp=0.4, sx=0.7, g0=2.0, g1=0.5, g2=0.3, sxi=0.5, somega=0.3, rho=0.3)
def obs():
    d0 = Design(seed=0); rng = np.random.default_rng(2026); data = simulate_market(d0, TRUE, "bertrand", rng)
    micro = data["micro"] + rng.normal(0, [0.03, 0.10]); return summaries(d0, data["s"], data["p"], micro)
def job(c):
    fn = f"results/solutions/ka/{c}.pkl"
    if os.path.exists(fn): return
    pickle.dump([one(c * 200 + i) for i in range(200)], open(fn, "wb"))
if __name__ == "__main__":
    os.makedirs("results/solutions/ka", exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "chunk": job(int(sys.argv[2])); print("chunk", sys.argv[2], "done")
