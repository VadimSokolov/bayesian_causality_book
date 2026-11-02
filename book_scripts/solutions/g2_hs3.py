import numpy as np
from scipy.special import expit
gh_x,gh_w=np.polynomial.hermite_e.hermegauss(40); gh_w=gh_w/gh_w.sum()
lb=np.linspace(-30,30,3001)  # log lambda grid
dl=lb[1]-lb[0]; lam=np.exp(lb); plam=2/(np.pi*(1+lam**2))*lam
b_abs=np.r_[np.exp(np.linspace(np.log(1e-5),np.log(.05),300)),np.linspace(.05+1e-3,45,6000)]
beta=np.r_[-b_abs[::-1],b_abs]
wb=np.gradient(beta)
def prior(g):
    out=np.zeros(len(b_abs))
    for i in range(0,len(b_abs)):
        s=lam*g; out[i]=np.sum(np.exp(-.5*(b_abs[i]/s)**2)/(s*np.sqrt(2*np.pi))*plam)*dl
    return np.r_[out[::-1],out]
def qf(l,sig):
    if sig==0: return expit(l)
    return (expit(l[:,None]+sig*gh_x)*gh_w).sum(1)
def post(g,a,M,S,sig,pr=None,flat=False):
    q=qf(a+beta,sig); ll=S*np.log(q)+(M-S)*np.log1p(-q); 
    w=wb*np.exp(ll-ll.max())*(1 if flat else pr)
    return np.sum(w*q)/np.sum(w)
gs=[10,1e6]; P={g:prior(g) for g in gs}
# whole estimator, chapter design
K,n,reps=20000,2000,200
for g in [10,1e6]:
  pr=P[g]; 
  for sig_use in [.3]:
    q=np.clip(qf(-0.0+beta,sig_use),1e-300,1-1e-16); lq=np.log(q); l1q=np.log1p(-q); memo={}
    def pm(M,S):
        if (M,S) not in memo:
            ll=S*lq+(M-S)*l1q; w=wb*pr*np.exp(ll-ll.max()); memo[(M,S)]=np.sum(w*q)/np.sum(w)
        return memo[(M,S)]
    r=np.random.default_rng(603)
    for J in [40,80,160]:
        d=[];e1=[];e2=[];emp=[]
        for _ in range(reps):
            e=r.uniform(.05,.95,K); th=expit(-1+6*(e-.5)+r.normal(0,.3,K));psi=th.mean()
            ed=np.quantile(e,np.linspace(0,1,J+1));ed[-1]+=1e-9;W=np.diff(np.searchsorted(np.sort(e),ed))/K
            X=r.integers(0,K,n);R=r.binomial(1,e[X]);Y=r.binomial(1,th[X]);o=R==1
            b=np.digitize(e[X[o]],ed)-1;Mj=np.bincount(b,minlength=J)[:J];Sj=np.bincount(b,weights=Y[o],minlength=J)[:J].astype(int)
            nz=Mj>0;hj=np.sum(W[nz]*Sj[nz]/Mj[nz])/np.sum(W[nz]);po=np.sum(W*np.array([pm(int(m),int(s)) for m,s in zip(Mj,Sj)]))
            d.append(po-hj);e1.append(po-psi);e2.append(hj-psi);emp.append((~nz).sum())
        d,e1,e2=map(np.array,(d,e1,e2))
        print('g',g,'J',J,'rms diff',round(float(np.sqrt((d**2).mean())),5),'mean diff',round(float(d.mean()),5),'rmse post',round(float(np.sqrt((e1**2).mean())),4),'rmse hj',round(float(np.sqrt((e2**2).mean())),4),'bias hj',round(float(e2.mean()),4),'empty',np.mean(emp),flush=True)
