# Instrumental variables chapter: Wright market simulation, Card (1995) college proximity, weak instrument study.
# The structural Bayesian posteriors (normal priors with variance 10^4 on the coefficients, inverse Wishart(3, 0.1 diag(var x, var y)) on the error
# covariance) are computed exactly for the coefficient tau (iv_exact.py integrates the other parameters out), and the posterior of the error
# correlation comes from the blocked Gibbs sampler of iv_exact.py, whose draws of tau agree with the exact posterior. Run from the repository root:
# python book_scripts/iv_card.py (before iv_weak2.py; both write results/res_iv.json). The check of the sampler that was used in earlier versions is iv_chain_check.py.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json, matplotlib, pandas as pd
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from iv_exact import exact_tau_cdf, cdf_quantiles, cdf_prob_greater, biv_block, split_rhat_ess
rng=np.random.default_rng(1928)
res={}
# 1. Wright: demand q = a - b p + u_d, supply q = c + d p + e*w + u_s, w a supply shifter (weather, yield)
n=200; b,d=1.0,0.8
w=rng.normal(0,1,n); ud=rng.normal(0,1,n); us=rng.normal(0,1,n)
p=(10-2+0*1 - 1.5*w + ud - us)/(b+d)   # equilibrium: a-bp+ud = c+dp+1.5w+us with a=10,c=2
q=10-b*p+ud
X=np.column_stack([np.ones(n),p])
ols=np.linalg.lstsq(X,q,rcond=None)[0][1]
iv=np.cov(w,q)[0,1]/np.cov(w,p)[0,1]
res['wright']={'ols_slope':float(ols),'iv_slope':float(iv),'true_demand_slope':-b,'true_supply_slope':d}
fig,ax=plt.subplots(figsize=(5.4,3.8))
# Gray scale print: the points run from light (small shifter) to black (large shifter), a map that is monotone in gray; the three lines differ by style and gray level, and the first two carry a thin white halo.
import matplotlib.patheffects as pe
halo=lambda lw: [pe.withStroke(linewidth=lw+1.4,foreground='white')]
sc=ax.scatter(p,q,c=w,cmap='Greys',vmin=w.min()-1,s=14,edgecolor='k',linewidth=0.3)
pp=np.linspace(p.min(),p.max(),10)
ax.plot(pp,10-b*pp,'-',color='k',lw=1.6,path_effects=halo(1.6),label='demand curve (truth)')
ax.plot(pp,q.mean()+ols*(pp-p.mean()),'--',color='0.55',lw=1.4,path_effects=halo(1.4),label=f'least squares, slope {ols:.2f}')
ax.plot(pp,q.mean()+iv*(pp-p.mean()),':',color='k',lw=2,label=f'instrumental variable, slope {iv:.2f}')
ax.set_xlabel('price'); ax.set_ylabel('quantity'); ax.legend(frameon=False,fontsize=8,handlelength=3)
cb=plt.colorbar(sc,ax=ax); cb.set_label('supply shifter (weather)')
plt.tight_layout(); plt.savefig('figs/fig_wright.pdf'); plt.close()
if '--figs-only' in sys.argv:   # redraw the Wright figure only (the seeded simulation above takes milliseconds); nothing is written to results/, and the slopes are compared with the saved ones
    print('Wright slopes now',res['wright'],'saved',json.load(open('results/res_iv.json')).get('wright')); sys.exit(0)

# 2. Card (1995)
D=pd.read_csv('data/card.csv').dropna(subset=['lwage','educ','nearc4'])
ctrl=['exper','expersq','black','smsa','south','smsa66']+[f'reg66{k}' for k in range(2,10)]
y=D['lwage'].values; x=D['educ'].values.astype(float); z=D['nearc4'].values.astype(float)
W=np.column_stack([np.ones(len(D))]+[D[c].values.astype(float) for c in ctrl])
N=len(y)
def ols_fit(Xm,yv):
    b=np.linalg.lstsq(Xm,yv,rcond=None)[0]; r=yv-Xm@b; s2=r@r/(len(yv)-Xm.shape[1])
    V=s2*np.linalg.inv(Xm.T@Xm); return b,np.sqrt(np.diag(V))
bo,so=ols_fit(np.column_stack([x,W]),y)
# first stage
Zf=np.column_stack([z,W]); bf,sf=ols_fit(Zf,x); F=(bf[0]/sf[0])**2
xh=Zf@bf
# 2SLS with correct se
X2=np.column_stack([x,W]); Xh=np.column_stack([xh,W])
b2=np.linalg.lstsq(Xh,y,rcond=None)[0]; r2=y-X2@b2; s2=r2@r2/(N-X2.shape[1])
V2=s2*np.linalg.inv(Xh.T@Xh); se2=np.sqrt(V2[0,0])
res['card']={'n':int(N),'ols':float(bo[0]),'ols_se':float(so[0]),'first_stage':float(bf[0]),'first_stage_se':float(sf[0]),'F':float(F),'tsls':float(b2[0]),'tsls_se':float(se2)}
# Structural Bayesian IV: exact posterior of the return to schooling, and the blocked Gibbs sampler for the error correlation
Zc=z[:,None]
tg,dens,cdf=exact_tau_cdf(y,x,Zc,W,scale=0.5,center=float(b2[0]))
q05,q50,q95=cdf_quantiles(tg,cdf,[0.05,0.5,0.95])
trap=getattr(np,'trapezoid',None) or np.trapz
mean_t=float(trap(tg*dens,tg)); sd_t=float(np.sqrt(trap((tg-mean_t)**2*dens,tg)))
chains=[biv_block(y,x,Zc,W,iters=11000,burn=1000,seed=s) for s in range(1,5)]
dr=np.concatenate(chains)
rh_t,ess_t=split_rhat_ess(np.array([c[:,0] for c in chains])); rh_r,ess_r=split_rhat_ess(np.array([c[:,2] for c in chains]))
res['card']['bayes']={'mean':mean_t,'sd':sd_t,'q05':float(q05),'q50':float(q50),'q95':float(q95),
  'p_gt_ols':cdf_prob_greater(tg,cdf,float(bo[0])),'p_pos':cdf_prob_greater(tg,cdf,0.0),'rho_mean':float(dr[:,2].mean()),
  'rho_q05':float(np.quantile(dr[:,2],0.05)),'rho_q95':float(np.quantile(dr[:,2],0.95)),'rho_p_neg':float((dr[:,2]<0).mean()),
  'sampler_q05':float(np.quantile(dr[:,0],0.05)),'sampler_q50':float(np.quantile(dr[:,0],0.5)),'sampler_q95':float(np.quantile(dr[:,0],0.95)),
  'sampler_chains':4,'sampler_iterations':11000,'sampler_burn_in':1000,'sampler_rhat_tau':rh_t,'sampler_ess_tau':ess_t,'sampler_rhat_rho':rh_r,'sampler_ess_rho':ess_r}
fig,ax=plt.subplots(1,2,figsize=(6.4,2.9))
ax[0].fill_between(tg,dens,color='0.6',alpha=0.8); ax[0].plot(tg,dens,color='k',lw=0.8)
ax[0].set_xlim(bo[0]-0.06,b2[0]+0.3)
ax[0].axvline(bo[0],color='k',ls='--',lw=1); ax[0].axvline(b2[0],color='k',ls=':',lw=1.5)
ax[0].set_xlabel('return to a year of schooling'); ax[0].set_yticks([])
ax[0].text(bo[0],ax[0].get_ylim()[1]*0.92,' OLS',fontsize=8); ax[0].text(b2[0],ax[0].get_ylim()[1]*0.8,' 2SLS',fontsize=8)
ax[1].hist(dr[:,2],bins=60,density=True,color='0.6',edgecolor='k',linewidth=0.3); ax[1].set_xlabel(r'error correlation $\rho$'); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig('figs/fig_card.pdf'); plt.close()

# 3. Weak instrument study in the spirit of quarter of birth
def sim(R=300,n=2000,pi=0.03,tau=0.1,rho=0.6,k=1,seed=7,scales=()):
    r=np.random.default_rng(seed); out=[]; sens={m:[] for m in scales}
    for rep in range(R):
        Z=r.normal(size=(n,k)); U=r.multivariate_normal([0,0],[[1,rho],[rho,1]],n)
        X=Z@np.full(k,pi)+U[:,0]; Y=tau*X+U[:,1]
        bo=np.polyfit(X,Y,1)[0]
        Zc=np.column_stack([Z,np.ones(n)]); xh=Zc@np.linalg.lstsq(Zc,X,rcond=None)[0]
        bt=np.cov(xh,Y)[0,1]/np.cov(xh,X)[0,1]
        fs=np.linalg.lstsq(Zc,X,rcond=None); rss=np.sum((X-xh)**2); F=((np.sum((xh-X.mean())**2))/k)/(rss/(n-k-1))
        tgrid,_,cdfw=exact_tau_cdf(Y,X,Z,np.ones((n,1)))
        lo,me,hi=cdf_quantiles(tgrid,cdfw,[0.05,0.5,0.95])
        out.append((bo,bt,me,lo<=tau<=hi,hi-lo,F))
        if rep<100:
            for m in scales:
                tg2,_,c2=exact_tau_cdf(Y,X,Z,np.ones((n,1)),s0_mult=m,scale=30.0,umax=5.0,G=24001)
                l2,m2,h2=cdf_quantiles(tg2,c2,[0.05,0.5,0.95]); sens[m].append((m2,l2<=tau<=h2,h2-l2))
    return np.array(out),{m:np.array(v) for m,v in sens.items()}
S,sens=sim(k=1,scales=(0.01,0.1,1.0,10.0)); M,_=sim(k=20,pi=0.03/np.sqrt(20)*1.0,seed=11)
def summ(S):
    return {'ols_med':float(np.median(S[:,0])),'tsls_med':float(np.median(S[:,1])),'bayes_med':float(np.median(S[:,2])),
            'bayes_cover':float(S[:,3].mean()),'bayes_len_med':float(np.median(S[:,4])),'F_med':float(np.median(S[:,5])),
            'bayes_med_iqr':[float(np.quantile(S[:,2],0.25)),float(np.quantile(S[:,2],0.75))],
            'tsls_iqr':[float(np.quantile(S[:,1],0.25)),float(np.quantile(S[:,1],0.75))]}
res['weak_one']=summ(S); res['weak_many']=summ(M)
res['weak_one']['prior_scale']={str(m):{'bayes_med':float(np.median(v[:,0])),'bayes_cover':float(v[:,1].mean()),'bayes_len_med':float(np.median(v[:,2])),'replications':int(len(v))} for m,v in sens.items()}
plt.rcParams.update({'font.size':9})
fig,ax=plt.subplots(2,1,figsize=(5.6,5.6),sharex=True)
for a,Sx,t in [(ax[0],S,'one weak instrument'),(ax[1],M,'twenty weak instruments')]:
    a.hist(np.clip(Sx[:,1],-1,1.2),bins=50,range=(-1,1.2),alpha=0.6,color='gray',label='2SLS',density=True)
    a.hist(np.clip(Sx[:,2],-1,1.2),bins=50,range=(-1,1.2),histtype='step',color='k',linewidth=1.2,label='posterior median',density=True)
    a.axvline(0.1,color='k',lw=1,label='true value'); a.axvline(np.median(Sx[:,0]),color='k',ls='--',lw=1,label='least squares')
    a.set_title(t,fontsize=9); a.set_xlabel(r'estimate of $\tau = 0.1$')
ax[1].legend(frameon=False,fontsize=8,loc='upper left'); ax[0].set_yticks([]); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig('figs/fig_weakiv.pdf'); plt.close()
import os
old=json.load(open('results/res_iv.json')) if os.path.exists('results/res_iv.json') else {}
old.update(res); json.dump(old,open('results/res_iv.json','w'),indent=1)
print(json.dumps(res,indent=1))
