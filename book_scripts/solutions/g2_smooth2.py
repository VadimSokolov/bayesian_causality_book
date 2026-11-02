import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit
rng=np.random.default_rng(607); K,n,reps=20000,2000,200
x=(np.arange(K)+.5)/K
J=13;deg=3;t=np.r_[[0]*deg,np.linspace(0,1,J-deg+1),[1]*deg]
Bm=BSpline.design_matrix(x,t,deg).toarray()[:,:J]; sub=np.arange(20,K,40); Bs=Bm[sub]
th=expit(-1+3*(2*x-1)+np.sin(2*np.pi*x))
def fit(D,Y,pv):
    b=np.zeros(J)
    for it in range(50):
        pr=expit(D@b); H=(D*(pr*(1-pr))[:,None]).T@D+np.eye(J)/pv; st=np.linalg.solve(H,D.T@(Y-pr)-b/pv); b=b+st
        if np.abs(st).max()<1e-8: break
    return b,H
out=[]
for r in range(reps):
    e=np.clip(expit(3*(2*x-1)+rng.normal(size=K)),.05,.95);X=rng.integers(0,K,n);R=rng.binomial(1,e[X]);Y=rng.binomial(1,th[X]);o=R==1;psi=th.mean();row=[]
    for sd in [10.,3.,1.5]:
        b,H=fit(Bm[X[o]],Y[o],sd**2);L=np.linalg.cholesky(np.linalg.inv(H));d=b[None]+rng.normal(size=(400,J))@L.T;ps=expit(d@Bs.T).mean(1)
        row+=[expit(Bs@b).mean()-psi,ps.mean()-psi,float(np.quantile(ps,.025)<=psi<=np.quantile(ps,.975))]
    out.append(row)
a=np.array(out);print('sd10,3,1.5: mode err, mean err, cov');print('bias',a.mean(0).round(4));print('rmse',np.sqrt((a**2).mean(0)).round(4))
