import numpy as np, sys
rng=np.random.default_rng(2204); B,n,reps=1000,100,5000
def run(p,thetaf,Js):
    res={J:[] for J in Js}
    for r in range(reps):
        th=thetaf(); psi=th.mean()
        X=rng.integers(0,B,n);R=rng.binomial(1,p[X]);Y=rng.binomial(1,th[X]);px=p[X]; li=(np.sum(R*Y)+1)/(np.sum(R)+2)
        for J in Js:
            edges=np.linspace(p.min(),p.max()+1e-9,J+1); bins=np.digitize(px,edges)-1; num=den=0.
            for j in range(J):
                s=bins==j
                if s.sum()==0: continue
                pt=px[s].mean(); num+=np.sum(R[s]*Y[s])/pt; den+=np.sum(R[s])/pt
            bs=num/den if den>0 else li
            wj=np.bincount(np.digitize(p,edges)-1,minlength=J)[:J]/B
            N=np.bincount(bins,minlength=J)[:J]; o=R==1
            M=np.bincount(bins[o],minlength=J)[:J]; S=np.bincount(bins[o],weights=Y[o],minlength=J)[:J]
            mu=(1+S)/(2+M); dir1=np.sum((1+N)/(J+n)*mu); dirh=np.sum((.5+N)/(J/2+n)*mu); known=np.sum(wj*mu)
            res[J].append([bs-psi,dir1-psi,dirh-psi,known-psi])
    for J in Js:
        E=np.array(res[J]);S=100*E**2
        print(J,'mse100 bs,dir1,dir.5,known',S.mean(0).round(3),'se',(S.std(0,ddof=1)/np.sqrt(reps)).round(3),'bias',E.mean(0).round(4),'paired se dir1-bs',round(float((S[:,1]-S[:,0]).std(ddof=1)/np.sqrt(reps)),3))
p2=np.linspace(.1,.9,B)
print('design 5');run(p2,lambda:.5*p2+.5*rng.uniform(0,1,B),[2,5,10,20,40])
print('off centre');run(p2,lambda:.25*p2+.25*rng.uniform(0,1,B),[2,5,10,20,40])
