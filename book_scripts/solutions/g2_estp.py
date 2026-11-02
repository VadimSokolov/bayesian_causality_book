import numpy as np
from scipy.special import expit
rng=np.random.default_rng(605);K,n,J,reps=50000,5000,20,6000
x=rng.normal(size=(K,2));c=np.column_stack([np.ones(K),x]);g0=np.array([0.,.8,.8])
e=expit(c@g0);th=expit(-.5+.5*(x[:,0]+x[:,1])+1.0*(x[:,0]-x[:,1]));psi=th.mean()
# population quantities
idx=np.argsort(c@g0);bins=np.array_split(idx,500);m=np.zeros(K)
for b in bins:m[b]=np.sum(e[b]*th[b])/np.sum(e[b])
A=-np.mean(((th-m)*(1-e))[:,None]*c,0);I=(c*(e*(1-e))[:,None]).T@c/K;ApIA=A@np.linalg.solve(I,A)
print('psi',psi,'A',A.round(4),'AIA',ApIA)
def est(gam,Xo,Yo):
    u=c@gam;ed=np.quantile(u,np.linspace(0,1,J+1));ed[0]-=1e-9;ed[-1]+=1e-9
    b=np.digitize(c[Xo]@gam,ed)-1;M=np.bincount(b,minlength=J)[:J];S=np.bincount(b,weights=Yo,minlength=J)[:J]
    return np.mean(S/np.maximum(M,1)),np.sum(S/np.maximum(M,1)*(1-S/np.maximum(M,1))/np.maximum(M,1))/J**2
def mle(D,R):
    g=np.zeros(3)
    for _ in range(25):
        pr=expit(D@g);H=(D*(pr*(1-pr))[:,None]).T@D;st=np.linalg.solve(H,D.T@(R-pr));g+=st
        if np.abs(st).max()<1e-9:break
    return g,H
rows=[]
for r in range(reps):
    X=rng.integers(0,K,n);R=rng.binomial(1,e[X]);Y=rng.binomial(1,th[X]);o=R==1;Xo,Yo=X[o],Y[o]
    gh,H=mle(c[X],R);p0,v0=est(g0,Xo,Yo);p1,v1=est(gh,Xo,Yo)
    between=np.nan
    if r<200:
        L=np.linalg.cholesky(np.linalg.inv(H));dr=[est(gh+L@rng.normal(size=3),Xo,Yo)[0] for _ in range(40)];between=np.var(dr,ddof=1)
    rows.append([p0,p1,v0,v1,between])
a=np.array(rows)
print('n*Var known',n*a[:,0].var(ddof=1),'n*Var est',n*a[:,1].var(ddof=1),'n*Var diff',n*(a[:,1]-a[:,0]).var(ddof=1),'predicted drop AIA',ApIA)
print('bias known',a[:,0].mean()-psi,'bias est',a[:,1].mean()-psi,'n*within',n*a[:,3].mean(),'n*between',n*np.nanmean(a[:,4]))
print('se n*Var',n*a[:,0].var(ddof=1)*np.sqrt(2/reps))
sdplug=np.sqrt(a[:,3]);sdcut=np.sqrt(a[:,3]+ApIA/n)
print('cov plug',np.mean(np.abs(a[:,1]-psi)<1.96*sdplug),'cov cut',np.mean(np.abs(a[:,1]-psi)<1.96*sdcut))
