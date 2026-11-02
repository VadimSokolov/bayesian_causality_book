# figure drawn by hs_theory_plot.py from results/res_hs_theory.json (October 2026 rerun: 500 replications, error bars of one Monte Carlo s.e., and a coverage panel pair)
import json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9})
res=json.load(open('results/res_hs_theory.json'))
fig,ax=plt.subplots(2,2,figsize=(6.4,5.4))
P=[d['p'] for d in res['quad']]
ax[0,0].errorbar(P,[d['risk']/d['plog'] for d in res['quad']],yerr=[d['se']/d['plog'] for d in res['quad']],fmt='o-',color='#b22222',label='horseshoe posterior mean')
ax[0,0].axhline(2,color='k',ls='--',lw=1,label='asymptotic minimax constant')
ax[0,0].set_xscale('log'); ax[0,0].set_ylim(0,2.3); ax[0,0].set_xlabel('number of nonzero means $p_n$'); ax[0,0].set_ylabel(r'risk / $p_n\log(n/p_n)$'); ax[0,0].legend(frameon=False,loc='lower right',fontsize=7)
Pp=[d['p'] for d in res['pred']]
ax[0,1].errorbar(Pp,[d['kl']/d['plog'] for d in res['pred']],yerr=[d['se']/d['plog'] for d in res['pred']],fmt='o-',color='#b22222',label='horseshoe predictive density')
ax[0,1].axhline(0.5,color='k',ls='--',lw=1,label=r'constant $1/(1+v_y/v_x)$')
ax[0,1].set_xscale('log'); ax[0,1].set_ylim(0,0.6); ax[0,1].set_xlabel('number of nonzero means $p_n$'); ax[0,1].set_ylabel(r'KL risk / $p_n\log(n/p_n)$'); ax[0,1].legend(frameon=False,loc='lower right',fontsize=7)
mk={5:'o',20:'s',80:'^'}; gr={5:'0.0',20:'0.4',80:'0.65'}
for d in res['unc']:
    a=[r['a'] for r in d['rows']]; p=d['p']
    ax[1,0].errorbar(a,[r['ball_L1'] for r in d['rows']],yerr=[r['ball_L1_se'] for r in d['rows']],marker=mk[p],ls='-',color=gr[p],ms=3,lw=1,label=f'$L=1$, $p_n={p}$')
    ax[1,0].errorbar(a,[r['ball_L2'] for r in d['rows']],yerr=[r['ball_L2_se'] for r in d['rows']],marker=mk[p],ls='--',color=gr[p],ms=3,lw=1,label=f'$L=2$, $p_n={p}$')
    ax[1,1].errorbar(a,[r['marginal'] for r in d['rows']],yerr=[r['marginal_se'] for r in d['rows']],marker=mk[p],ls='-',color=gr[p],ms=3,lw=1,label=f'$p_n={p}$')
ax[1,0].axhline(0.95,color='#b22222',ls=':',lw=1); ax[1,1].axhline(0.95,color='#b22222',ls=':',lw=1)
ax[1,0].set_ylim(0,1.03); ax[1,1].set_ylim(0,1.03)
ax[1,0].set_xlabel(r'signal size $a$ (nonzero means $a\sqrt{2\log(n/p_n)}$)'); ax[1,0].set_ylabel('coverage of the 95% credible ball'); ax[1,0].legend(frameon=False,fontsize=6,ncol=2,loc='lower right')
ax[1,1].set_xlabel(r'signal size $a$'); ax[1,1].set_ylabel('coverage of 95% intervals, nonzero means'); ax[1,1].legend(frameon=False,fontsize=7,loc='lower right')
plt.tight_layout(); plt.savefig('figs/fig_hs_theory.pdf')
