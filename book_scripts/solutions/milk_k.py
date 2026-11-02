import numpy as np, json
s = json.load(open('results/res_milk_unrecorded.json'))['sensitivity_r0.8']
th, se, aS, bS = s['tau_hat'], s['se'], s['a_S'], s['b_S']
print(th, se, aS, bS, aS*bS, 'k explain', np.sqrt(th/(aS*bS)), 'k truth', np.sqrt((th-.2)/(aS*bS)), s['a_U']/aS, s['b_U']/bS)
z = np.random.default_rng(1933).normal(size=(3, 200000))
for k in (.25,.5,1.):
    t = th - k*aS*(1+z[0])*k*bS*(1+z[1]) + se*z[2]; print(k, t.mean().round(3), np.quantile(t,[.025,.975]).round(3), (t>0).mean().round(4), 'sd ab', (np.sqrt(3)*k*k*aS*bS).round(3))
rng = np.random.default_rng(1934)
r = rng.uniform(.5,.95,200000); k = np.sqrt(1-r**2)
print('k range', np.sqrt(1-.25), np.sqrt(1-.95**2), 'mean k', k.mean())
t = th - k*aS*(1+z[0])*k*bS*(1+z[1]) + se*z[2]; print('mix', t.mean().round(3), np.quantile(t,[.025,.975]).round(3), (t>0).mean().round(4), (t<.2).mean().round(3))
d=json.load(open('results/res_did_sim2.json'))['N200_dep0.1_a']['hs']; print({a:(round(b,4) if isinstance(b,float) else b) for a,b in d.items()})
