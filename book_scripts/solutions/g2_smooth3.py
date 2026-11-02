import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit
rng=np.random.default_rng(606); K,n=20000,2000
x=(np.arange(K)+.5)/K
J=13;deg=3;t=np.r_[[0]*deg,np.linspace(0,1,J-deg+1),[1]*deg]
Bm=BSpline.design_matrix(x,t,deg).toarray()[:,:J]; sub=np.arange(20,K,40); Bs=Bm[sub]; pv=100.;df=6;Nd=1500
def fit(D,Y):
    b=np.zeros(J)
    for it in range(50):
        pr=expit(D@b); H=(D*(pr*(1-pr))[:,None]).T@D+np.eye(J)/pv; st=np.linalg.solve(H,D.T@(Y-pr)-b/pv); b=b+st
        if np.abs(st).max()<1e-8: break
    return b,H
def post(D,Y):
    b,H=fit(D,Y);L=np.linalg.cholesky(np.linalg.inv(H));Zs=rng.normal(size=(Nd,J));c2=rng.chisquare(df,Nd)/df
    Rv=Zs/np.sqrt(c2)[:,None];d=b+Rv@L.T
    eta=d@D.T;ll=(Y*eta-np.logaddexp(0,eta)).sum(1)-(d**2).sum(1)/(2*pv)
    lq=-(df+J)/2*np.log1p((Rv**2).sum(1)/df);lw=ll-lq;w=np.exp(lw-lw.max());w/=w.sum()
    ps=expit(d@Bs.T).mean(1);o=np.argsort(ps);cw=np.cumsum(w[o]);lo,hi=ps[o][np.searchsorted(cw,.025)],ps[o][np.searchsorted(cw,.975)]
    return np.sum(w*ps),lo,hi,1/np.sum(w**2),expit(Bs@b).mean()
def rep(design):
    if design=='smooth':
        e=np.clip(expit(3*(2*x-1)+rng.normal(size=K)),.05,.95);th=expit(-1+3*(2*x-1)+np.sin(2*np.pi*x))
    else:
        e=rng.uniform(.05,.95,K);th=expit(-1+6*(e-.5)+rng.normal(0,.3,K))
    X=rng.integers(0,K,n);R=rng.binomial(1,e[X]);Y=rng.binomial(1,th[X]);psi=th.mean();o=R==1;Xo,Yo=X[o],Y[o]
    N=np.bincount(Xo,minlength=K);S=np.bincount(Xo,weights=Yo,minlength=K)
    ht=np.mean(R*Y/e[X]);ind=((1+S)/(2+N)).mean();li=(Yo.sum()+1)/(o.sum()+2)
    m,lo,hi,ess,mode=post(Bm[Xo],Yo.astype(float))
    return [ht-psi,ind-psi,li-psi,m-psi,mode-psi,float(lo<=psi<=hi),hi-lo,ess,psi]
for d,reps in [('rough',200)]:
    a=np.array([rep(d) for _ in range(reps)]);E=a[:,:5]
    print(d,reps,'HT ind li mean mode');print(' bias',E.mean(0).round(4),'se',(E.std(0,ddof=1)/np.sqrt(reps)).round(4));print(' rmse',np.sqrt((E**2).mean(0)).round(4));print(' cov',a[:,5].mean(),'width',a[:,6].mean().round(4),'min/med ess',np.quantile(a[:,7],[.0,.5]).round(0),'psi',a[:,8].mean().round(4),flush=True)
