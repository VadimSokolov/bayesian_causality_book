import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit,logit
rng=np.random.default_rng(606); K,n,reps=20000,2000,200
x=(np.arange(K)+.5)/K
J=13;deg=3;t=np.r_[[0]*deg,np.linspace(0,1,J-deg+1),[1]*deg]
Bm=BSpline.design_matrix(x,t,deg).toarray(); Bm=Bm[:,:J]
sub=np.arange(20,K,40); Bs=Bm[sub]; pv=10.0**2
def smooth(Xi,Yi):
    D=Bm[Xi]; beta=np.zeros(J)
    for it in range(50):
        pr=expit(D@beta); g=D.T@(Yi-pr)-beta/pv; W=pr*(1-pr); H=(D*W[:,None]).T@D+np.eye(J)/pv
        st=np.linalg.solve(H,g); beta=beta+st
        if np.abs(st).max()<1e-8: break
    L=np.linalg.cholesky(np.linalg.inv(H)); dr=beta[None,:]+rng.normal(size=(400,J))@L.T
    ps=expit(dr@Bs.T).mean(1); return ps.mean(),np.quantile(ps,.025),np.quantile(ps,.975)
def rep(design):
    if design=='smooth':
        e=np.clip(expit(3*(2*x-1)+rng.normal(size=K)),.05,.95); th=expit(-1+3*(2*x-1)+np.sin(2*np.pi*x))
    else:
        e=rng.uniform(.05,.95,K); th=expit(-1+6*(e-.5)+rng.normal(0,.3,K))
    X=rng.integers(0,K,n);R=rng.binomial(1,e[X]);Y=rng.binomial(1,th[X]);psi=th.mean()
    ht=np.mean(R*Y/e[X]); o=R==1; Xo,Yo=X[o],Y[o]
    N=np.bincount(Xo,minlength=K);S=np.bincount(Xo,weights=Yo,minlength=K)
    ind=((1+S)/(2+N)).mean(); li=(Yo.sum()+1)/(o.sum()+2)
    sm,lo,hi=smooth(Xo,Yo)
    return [ht-psi,ind-psi,li-psi,sm-psi],lo<=psi<=hi,hi-lo,psi,np.corrcoef(e,th)[0,1],o.sum()
for d in ['smooth','rough']:
    A=[rep(d) for _ in range(reps)]
    E=np.array([a[0] for a in A])
    print(d,'names HT indep li smooth','bias',E.mean(0).round(4),'rmse',np.sqrt((E**2).mean(0)).round(4),'cov',np.mean([a[1] for a in A]),'width',round(float(np.mean([a[2] for a in A])),4),'psi',round(float(np.mean([a[3] for a in A])),4),'corr',round(float(np.mean([a[4] for a in A])),3),'sel',np.mean([a[5] for a in A]),'mcse rmse2', (E**2).std(0,ddof=1).round(5)/np.sqrt(reps))
