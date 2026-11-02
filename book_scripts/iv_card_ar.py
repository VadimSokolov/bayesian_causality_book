# Anderson Rubin confidence sets for the Card (1995) college proximity example (Chapter 12, Instruments).
# Same sample and controls as iv_card.py and iv_weak2.py. The reduced form posterior of iv_weak2.py (flat prior on the coefficients,
# Jeffreys prior on the error covariance) has an equal tailed interval that coincides with the Anderson Rubin set when the first stage is strong.
# Run from the repository root: python book_scripts/iv_card_ar.py. Writes results/res_iv_ar.json. No random numbers are used.
import numpy as np, pandas as pd, json
from scipy import stats

D = pd.read_csv('data/card.csv').dropna(subset=['lwage', 'educ', 'nearc4'])
ctrl = ['exper', 'expersq', 'black', 'smsa', 'south', 'smsa66'] + [f'reg66{k}' for k in range(2, 10)]
y = D['lwage'].values; x = D['educ'].values.astype(float); z = D['nearc4'].values.astype(float)
W = np.column_stack([np.ones(len(y))] + [D[c].values.astype(float) for c in ctrl])
n = len(y); k = W.shape[1] + 1
P = lambda v: v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
yt, xt, zt = P(y), P(x), P(z)
szz = zt @ zt
a = (zt @ yt) / szz                      # reduced form coefficient of the instrument in the wage equation
b = (zt @ xt) / szz                      # first stage coefficient of the instrument in the schooling equation
ry, rx = yt - a * zt, xt - b * zt
sxx, sxy, syy = rx @ rx / (n - k), rx @ ry / (n - k), ry @ ry / (n - k)

def ar_set(level):
    # tau0 is in the set when |a - tau0 b| <= c * sqrt((syy - 2 tau0 sxy + tau0^2 sxx) / szz)
    c = stats.t.ppf(1 - (1 - level) / 2, n - k)
    q2 = b ** 2 - c ** 2 * sxx / szz
    q1 = -2 * a * b + 2 * c ** 2 * sxy / szz
    q0 = a ** 2 - c ** 2 * syy / szz
    assert q2 > 0, 'unbounded set'
    r = np.sort(np.roots([q2, q1, q0]).real)
    return [float(r[0]), float(r[1])]

out = {'n': int(n), 'df': int(n - k), 'first_stage_t': float(b / np.sqrt(sxx / szz)), 'ar90': ar_set(0.90), 'ar95': ar_set(0.95)}
json.dump(out, open('results/res_iv_ar.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
