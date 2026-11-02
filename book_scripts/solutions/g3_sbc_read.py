import json, numpy as np
from scipy import stats
res=[json.loads(l) for l in open('results/solutions/sbc_t100_b5000.jsonl')]
print(len(res))
for name,sel in [('r<2000',lambda r:r<2000),('r>=2000',lambda r:r>=2000),('all',lambda r:True)]:
    x=[a for a in res if sel(a['r'])]
    for k in ['rt','rg','rs']:
        rk=np.array([a[k] for a in x]); c=np.bincount(rk,minlength=20)
        z=(rk.mean()-9.5)/(np.sqrt(399/12)/np.sqrt(len(rk)))
        co=[int(c[i:i+4].sum()) for i in range(0,20,4)]
        print(name,k,len(rk),'chi %.2f p %.3f'%stats.chisquare(c),'mean %.3f z %.2f'%(rk.mean(),z),'extreme %.3f'%np.mean((rk==0)|(rk==19)),co, c.tolist())
