"""Calibration of the pair model posterior of ex:chframeworks:8: m=50, kappa=sigma=1, tau=1, true lam=.9; 400 data sets, seed 20261102."""
import numpy as np
from scipy import stats
s=2.0; al=1.0; m=50
grid=np.linspace(-3,5,1601)
def post(D,l):
    z=D[None,:]-grid[:,None]
    lp=(stats.norm.logpdf(z,0,s)+np.log(l*stats.norm.cdf(al*z/s)+(1-l)*stats.norm.cdf(-al*z/s))).sum(1)
    w=np.exp(lp-lp.max()); w/=w.sum(); mu=(w*grid).sum(); c=np.cumsum(w)
    return mu,np.sqrt((w*(grid-mu)**2).sum()),grid[np.searchsorted(c,.025)],grid[np.searchsorted(c,.975)]
r=np.random.default_rng(20261102); out=[]
for k in range(400):
    V=r.normal(0,np.sqrt(2),m); sg=np.where(r.uniform(size=m)<.9,1,-1); D=1+sg*np.abs(V)+r.normal(0,np.sqrt(2),m)
    a=post(D,.5); b=post(D,.9)
    out.append((a[0],a[1],a[2]<=1<=a[3],b[0],b[1],b[2]<=1<=b[3]))
o=np.array(out,float)
print("lam=.5 mean %.3f sd %.3f cover %.3f | lam=.9 mean %.3f sd %.3f cover %.3f | gap mean %.3f sd %.3f"%(o[:,0].mean(),o[:,1].mean(),o[:,2].mean(),o[:,3].mean(),o[:,4].mean(),o[:,5].mean(),(o[:,0]-o[:,3]).mean(),(o[:,0]-o[:,3]).std()))
