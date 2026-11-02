import numpy as np,time
from scipy.special import expit
rng=np.random.default_rng(66)
z=np.array([[1,-1.5],[1,-.5],[1,0.],[1,.5],[1,1.5]]); M=np.array([3,1,2,0,0]);S=np.array([3,0,1,0,0]);V=M>0; v=4.
# importance sampling from prior
N=3_000_000
gam=rng.normal(0,np.sqrt(v),(N,2)); g=np.abs(rng.standard_cauchy(N)); lam=np.abs(rng.standard_cauchy((N,5)))
eta=rng.normal(size=(N,5))*lam*g[:,None]; th=expit(gam@z.T+eta)
lw=(S*np.log(np.maximum(th,1e-300))+(M-S)*np.log(np.maximum(1-th,1e-300))).sum(1); w=np.exp(lw); w/=w.sum()
print('IS ESS',1/np.sum(w**2)); psi=th.mean(1)
print('IS psi',np.sum(w*psi),'th',np.sum(w[:,None]*th,0).round(4),'gam',np.sum(w[:,None]*gam,0).round(4),'P(g>1)',np.sum(w*(g>1)))
T=60; K2=(np.arange(1,T+1)-.5)**2; K3=None
def rpg(b,c):
    a=K2+c**2/(4*np.pi**2); s=(rng.gamma(b,1,T)/a).sum()
    tail=b*np.sum(1/(K3**2+c**2/(4*np.pi**2))) if False else b/(T+0.0+1e-9)  # int_{T}^inf dk/(k-1/2)^2 approx 1/T, conservative for small c
    return (s+tail)/(2*np.pi**2)
Mv=M[V];kap=S[V]-Mv/2;zv=z[V];q=len(Mv);uv=~V
eta=np.zeros(q);gm=np.zeros(2);l2=np.ones(q);nu=np.ones(q);g2=1.;xi=1.
it=120000;burn=10000;out=[];t0=time.time()
ig=lambda a,b:1/rng.gamma(a,1/b)
for s in range(it):
    zeta=zv@gm+eta; om=np.array([rpg(Mv[i],zeta[i]) for i in range(q)])
    d=l2*g2; sd=1/om+d; yt=kap/om
    Pg=np.eye(2)/v+(zv/sd[:,None]).T@zv; Cg=np.linalg.inv(Pg); gm=rng.multivariate_normal(Cg@((zv/sd[:,None]).T@yt),Cg)
    pe=om+1/d; eta=(kap-om*(zv@gm))/pe+rng.normal(size=q)/np.sqrt(pe)
    l2=ig(1,1/nu+eta**2/(2*g2)); nu=ig(1,1+1/l2)
    g2=ig((q+1)/2,1/xi+np.sum(eta**2/l2)/2); xi=ig(1,1+1/g2)
    if s>=burn:
        lu=np.abs(rng.standard_cauchy(uv.sum())); eu=rng.normal(size=uv.sum())*lu*np.sqrt(g2)
        tv=expit(zv@gm+eta); tu=expit(z[uv]@gm+eu); t_=np.r_[tv,tu]
        out.append(np.r_[t_.mean(),t_,gm,g2>1])
out=np.array(out);print('time',time.time()-t0)
bm=out[:len(out)//100*100].reshape(100,-1,out.shape[1]).mean(1)
print('Gibbs psi,th,gam,P(g>1)',out.mean(0).round(4));print('batch se',(bm.std(0,ddof=1)/10).round(4))
