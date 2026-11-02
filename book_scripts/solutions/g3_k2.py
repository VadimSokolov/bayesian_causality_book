import numpy as np
from scipy import stats
LS=np.linspace(-40,25,6501);DL=LS[1]-LS[0];S=np.exp(LS)
def dens(g,plus):
    u=S/g
    if plus:
        lu=np.log(u)
        with np.errstate(all="ignore"):
            d=np.where(np.abs(lu)<1e-8,2/np.pi**2,4*lu/(np.pi**2*(u**2-1)))
        return d*u
    return 2/(np.pi*(1+u**2))*u
def sh(y,g,plus):
    y=np.atleast_1d(y);w=dens(g,plus);k=1/(1+S**2)
    like=np.exp(-.5*np.outer(y**2,k))*np.sqrt(k)
    return (like@(w*(1-k)))/(like@w)
ys=np.array([0,1,2,3,3.5,4,5,6,8.])
G=[(0.00716,False),(0.00152,True)]
for g,p in G: print(g,p,np.round(sh(ys,g,p),3))
gy=np.linspace(-6,20,1041);dy=gy[1]-gy[0]
for mu in [2,3,4,5,8.]:
    r=[mu-np.sum(sh(gy,g,p)*gy*stats.norm.pdf(gy-mu))*dy for g,p in G]
    print(mu,np.round(r,3))
for g in [0.00716,0.00152]:
    print(g,'HS null',2*g/np.pi,'HS+ null',4*g/np.pi**2*(np.log(1/g)-1),'escape',2/np.pi*np.arctan(g),4/np.pi**2*(np.log(1/g)+1)*g)
