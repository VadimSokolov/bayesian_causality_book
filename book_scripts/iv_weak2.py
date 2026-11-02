# Reduced form posterior of tau for one instrument (iid draws), and the structural posterior with a hierarchical normal prior on the twenty first
# stage coefficients (blocked Gibbs sampler of iv_exact.py, four chains per data set). Run from the repository root after iv_card.py.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json
from iv_exact import biv_block, split_rhat_ess
def rf_ratio(Y,X,Z,draws=4000,seed=0,k_extra=0):
    # reduced form posterior with flat prior on coefficients and Jeffreys prior on Sigma; tau = piY/piT (one instrument)
    # k_extra counts columns partialled out beforehand, so the Wishart draw has n - k - k_extra degrees of freedom
    r=np.random.default_rng(seed); n=len(Y)
    A=np.column_stack([Z,np.ones(n)]); k=A.shape[1]
    AtAi=np.linalg.inv(A.T@A); Bh=AtAi@A.T@np.column_stack([X,Y]); E=np.column_stack([X,Y])-A@Bh; S=E.T@E
    out=[]
    Lc=np.linalg.cholesky(AtAi)
    for d in range(draws):
        G=np.linalg.cholesky(np.linalg.inv(S)); Xs=(G@r.normal(size=(2,n-k-k_extra))).T; Sig=np.linalg.inv(Xs.T@Xs)
        B=Bh+Lc@r.normal(size=(k,2))@np.linalg.cholesky(Sig).T
        out.append(B[0,1]/B[0,0])
    return np.array(out)
def study(k,pi,R=200,n=2000,tau=0.1,rho=0.6,seed=7):
    r=np.random.default_rng(seed); out=[]; rhats=[]; esss=[]
    for rep in range(R):
        Z=r.normal(size=(n,k)); U=r.multivariate_normal([0,0],[[1,rho],[rho,1]],n)
        X=Z@np.full(k,pi)+U[:,0]; Y=tau*X+U[:,1]
        if k==1:
            d=rf_ratio(Y,X,Z,draws=3000,seed=rep)
        else:
            ch=[biv_block(Y,X,Z,np.ones((n,1)),iters=4000,burn=1000,seed=1000*rep+c,hier=True)[:,0] for c in range(4)]
            d=np.concatenate(ch); rh,es=split_rhat_ess(np.array(ch)); rhats.append(rh); esss.append(es)
        lo,me,hi=np.quantile(d,[0.05,0.5,0.95]); out.append((me,lo<=tau<=hi,hi-lo))
    o=np.array(out)
    summary={'med':float(np.median(o[:,0])),'cover':float(o[:,1].mean()),'len_med':float(np.median(o[:,2])),'len_q25':float(np.quantile(o[:,2],0.25))}
    if rhats:
        summary.update(chains=4,iterations=4000,burn_in=1000,rhat_max=float(max(rhats)),rhat_median=float(np.median(rhats)),share_rhat_below_1_01=float((np.array(rhats)<1.01).mean()),ess_min=float(min(esss)),ess_median=float(np.median(esss)))
    return summary
if '--card-only' in sys.argv: res={}  # the two simulations do not depend on the Card degrees of freedom; their saved values in results/res_iv.json are kept
else: res={'rf_one':study(1,0.03),'hier_many':study(20,0.03/np.sqrt(20),seed=11)}
# Card with reduced form ratio
import pandas as pd
D=pd.read_csv('data/card.csv').dropna(subset=['lwage','educ','nearc4'])
ctrl=['exper','expersq','black','smsa','south','smsa66']+[f'reg66{k}' for k in range(2,10)]
y=D['lwage'].values; x=D['educ'].values.astype(float); z=D['nearc4'].values.astype(float)
W=np.column_stack([D[c].values.astype(float) for c in ctrl])
# partial out controls, then reduced form ratio
Wc=np.column_stack([np.ones(len(y)),W]); P=lambda v: v-Wc@np.linalg.lstsq(Wc,v,rcond=None)[0]
# full design: nearc4, constant and 14 controls = 16 columns, so n - 16 degrees of freedom (k_extra = 14 beyond the two columns of the partialled design)
d=rf_ratio(P(y),P(x),P(z)[:,None],draws=20000,seed=3,k_extra=Wc.shape[1]-1)
q=np.quantile(d,[0.05,0.25,0.5,0.75,0.95])
res['card_rf']={'q05':float(q[0]),'q25':float(q[1]),'q50':float(q[2]),'q75':float(q[3]),'q95':float(q[4]),'p_pos':float((d>0).mean()),'p_gt_ols':float((d>0.0747).mean())}
old=json.load(open('results/res_iv.json')); old.update(res); json.dump(old,open('results/res_iv.json','w'),indent=1)
print(json.dumps(res,indent=1))
