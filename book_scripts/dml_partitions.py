# Chapters 5, 7 and 16: spread of DML 401(k) estimates over random fold partitions (20 lasso, 5 forest)
import sys,json,numpy as np
src=open("book_scripts/app401k_book.py").read()
head=src.split('import os\nif os.path.exists("results/res401k.json")')[0]
exec(head)
res={'lasso':[dml(P,'lasso',s) for s in range(20)],'rf':[dml(X,'rf',s) for s in range(5)]}
json.dump(res,open('results/res_dml_partitions.json','w'))
for k,v in res.items():
    th=np.array([a for a,b in v]); se=np.array([b for a,b in v])
    print(k, 'min',th.min().round(2),'max',th.max().round(2),'median',np.median(th).round(2),'sd',th.std(ddof=1).round(2),'se med',np.median(se).round(2), 'seed0',v[0])
