"""Resumable SBC driver. Usage: g3_sbc_batch.py THIN NREP BURN. Appends one JSON line per replication."""
import os; os.makedirs("results/solutions", exist_ok=True)
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import multiprocessing as mp
import g3_sbc as M

if __name__ == "__main__":
    thin, nrep, burn = map(int, sys.argv[1:4])
    fn = f"results/solutions/sbc_t{thin}_b{burn}.jsonl"
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["r"] for l in open(fn)}
    todo = [(r, burn, thin, 19) for r in range(nrep) if r not in done]
    if todo:
        with mp.Pool(8) as p, open(fn, "a") as f:
            for x in p.imap_unordered(M.rep_run, todo, chunksize=1):
                f.write(json.dumps(x) + "\n"); f.flush()
    res = [json.loads(l) for l in open(fn)]
    print("reps", len(res))
    if len(res) >= nrep:
        print(json.dumps(M.report(res, 19)))
