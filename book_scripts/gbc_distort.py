# Distorted expected utilities for two programmes from posterior predictive draws (Chapter on GBC)
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
rng=np.random.default_rng(2026); R=200000
# programme A: modest, reliable benefit; programme B: larger typical benefit, left tail of harm
A=rng.normal(1.0,0.35,R)
B=np.where(rng.random(R)<0.12, rng.normal(-0.9,0.5,R), rng.normal(1.35,0.35,R))
taus=(np.arange(4000)+0.5)/4000
QA,QB=np.quantile(A,taus),np.quantile(B,taus)
dist={
 'Risk neutral':lambda t:t,
 'CVaR 0.25':lambda t:0.25*t,
 'Wang, eta = -0.75':lambda t:norm.cdf(norm.ppf(t)-0.75),
 'CPW, eta = 0.71':lambda t:t**0.71/(t**0.71+(1-t)**0.71)**(1/0.71),
 'Power averse, eta = 1':lambda t:1-(1-t)**0.5,
 'Power seeking, eta = 1':lambda t:t**0.5,
}
def V(Q,b):
    return float(np.mean(np.interp(b(taus),taus,Q)))
res={k:(V(QA,b),V(QB,b)) for k,b in dist.items()}
res['mean']=(float(A.mean()),float(B.mean()))
res['P(B<0)']=float((B<0).mean())
json.dump(res,open('results/res_gbc.json','w'),indent=1)
# CVaR level at which the planner is indifferent between A and B (Exercise ex:chgbc:4)
from scipy.optimize import brentq
e_star=brentq(lambda e: V(QB,lambda t:e*t)-V(QA,lambda t:e*t),0.3,1.0)
print('CVaR indifference level',e_star)
for k,v in res.items(): print(k,v)
plt.rcParams.update({'font.size':10})
fig,ax=plt.subplots(2,1,figsize=(5.6,6.4))
ax[0].plot(taus,QA,label='Programme A',color='#2a78d6'); ax[0].plot(taus,QB,label='Programme B',color='#eb6834')
ax[0].axhline(0,color='gray',lw=0.6); ax[0].set_xlabel(r'quantile level $u$'); ax[0].set_ylabel(r'utility quantile $Q_{Z(a)}(u)$'); ax[0].legend(frameon=False)
names=[k for k in dist]; x=np.arange(len(names)); w=0.38
ax[1].bar(x-w/2,[res[k][0] for k in names],w,color='#2a78d6',label='Programme A')
ax[1].bar(x+w/2,[res[k][1] for k in names],w,color='#eb6834',label='Programme B')
ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(', ','\n').replace('Power ','Power\n').replace('eta',r'$\eta$') for n in names],fontsize=8)
ax[1].axhline(0,color='gray',lw=0.6); ax[1].set_ylabel(r'distorted value $V_\beta(a)$'); ax[1].legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig('figs/fig_gbc_distort.pdf'); print('saved')
