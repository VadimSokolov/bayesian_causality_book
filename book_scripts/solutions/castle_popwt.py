"""sol4 F4. Is popwt constant within state, and does it equal the state's mean population over 2000 to 2010, the population being implied by a crime count and rate when the file has both?"""
import pandas as pd
d = pd.read_stata('data/castle.dta')
cs = [c for c in d.columns if not c.startswith(('r20', 'trend', 'lead', 'lag', 'year', 'pre', 'post'))]
print(cs[:60])
g = d.groupby('sid'); print('within sd max', g.popwt.std().max())
print(g.popwt.first().sort_values().tail(3).round(0).to_dict(), g.popwt.first().sort_values().head(2).round(0).to_dict())
for c in [c for c in cs if c + '_c' in d.columns or (c.endswith('_c') and c[:-2] in d.columns)]:
    if c.endswith('_c'):
        p = d[c] / d[c[:-2]] * 1e5; r = (g.popwt.first() / p.groupby(d.sid).mean() - 1).abs(); print(c, 'median rel err', r.median(), 'max', r.max())
