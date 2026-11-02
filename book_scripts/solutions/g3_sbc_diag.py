import json, sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3_sbc as M
res=[json.loads(l) for l in open('results/solutions/sbc_t100_b5000.jsonl')]
rows=[]
for a in res:
    rng=np.random.default_rng([20261105,a['r']])
    s2,g,lam,tau,beta=M.draw_prior(rng)
    Y=M.Z@np.r_[tau,beta]+np.sqrt(s2)*rng.standard_normal(M.N)
    rows.append((a['r'],a['rs'],a['rt'],a['rg'],np.log10(Y@Y),np.log10(s2),np.abs(beta).max()/np.sqrt(s2),g))
R=np.array(rows)
assert abs(R[R[:,0]==0][0,7]-[a for a in res if a['r']==0][0]['g0'])<1e-12
yy=R[:,4]
print('log10 YY quantiles',np.round(np.quantile(yy,[0,.1,.25,.5,.75,.9,.99,1]),1))
edges=[-9,2,3,4,6,9,12,99]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(yy>=lo)&(yy<hi)
    if m.sum()==0: continue
    for c,nm in [(1,'s2'),(2,'tau'),(3,'g')]:
        rk=R[m,c]; z=(rk.mean()-9.5)/(np.sqrt(399/12)/np.sqrt(m.sum()))
        print(f'YY 1e{lo}-1e{hi} n={m.sum()} {nm} mean {rk.mean():.2f} z {z:.2f} top {np.mean(rk==19):.3f} bottom {np.mean(rk==0):.3f}')
