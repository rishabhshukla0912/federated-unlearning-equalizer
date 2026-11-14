import csv,glob,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','xtick.major.size':2.5,'ytick.major.size':2.5,'ytick.right':True,'mathtext.fontset':'stix'})
import pathlib
HERE=pathlib.Path(__file__).resolve().parent
R=str(HERE.parent/'results'/'protocol_v2')+'/'
(HERE/'out').mkdir(exist_ok=True)
rows=[]
for f in sorted(glob.glob(R+'runs/confirm_*/summary.csv')):
    rows+=[r for r in csv.DictReader(open(f)) if r['split']=='test']
def get(inj,m,c): return {int(r['seed']):float(r[c]) for r in rows if r['inject']==inj and r['method']==m and r[c]!=''}
M=[('FedAvg-all','All-client'),('FedAvg-retained (gold)','Gold'),('FedEraser','FedEraser'),('Grad-ascent','Grad. ascent'),('Retain-finetune','Fine-tune')]
grid=list(csv.DictReader(open(R+'tost_margin_grid.csv')))
fig,(a,b)=plt.subplots(1,2,figsize=(7.16,2.35),gridspec_kw={'width_ratios':[1,1.35]})
x=np.arange(len(M))
for i,(k,l) in enumerate(M):
    v=np.array(list(get('canary',k,'canary_asr').values()))
    a.bar(i,v.mean(),0.62,color={'All-client':'#8c8c8c','Gold':'#ffffff','FedEraser':'#3b6ea5','Grad. ascent':'#d4d4d4','Fine-tune':'#5a5a5a'}[l],hatch='////' if l=='Grad. ascent' else '',edgecolor='black',lw=0.6,zorder=2)
    a.errorbar(i,v.mean(),v.std(ddof=1),color='black',capsize=2,lw=0.7,zorder=3)
    a.plot(np.full(len(v),i+0.2),v,'o',ms=1.8,color='black',zorder=4)
a.axhline(90,ls='--',lw=0.7,color='black',zorder=1)
a.set_xticks(x);a.set_xticklabels(['All-\nclient','Gold','Fed-\nEraser','Grad.\nascent','Fine-\ntuning'],fontsize=7);a.set_ylabel('Canary ASR (%)');a.set_ylim(0,105)
a.set_title('(a)',loc='left',fontsize=8,pad=3);a.set_xlim(-0.55,4.55)

eps=[('canary','canary_asr','Canary ASR'),('canary','retain_acc','Canary retained'),('canary','forget_acc','Canary forget'),('iq','forget_acc','IQ forget'),('iq','retain_acc','IQ retained')]
meths=[('FedEraser','FedEraser','#3b6ea5','o'),('Grad-ascent','Gradient ascent','0.55','^'),('Retain-finetune','Fine-tuning','black','v')]
for d,al in [(3,0.10),(2,0.13),(1.5,0.16),(1,0.2)]: b.axvspan(-d,d,color='#3b6ea5',alpha=al*0.7,lw=0)
b.axvline(0,color='0.2',lw=0.5)
y=0;yt=[];yl=[]
for inj,c,lab in eps:
    if c=='canary_asr': continue
    for j,(mk,ml,col,mkr) in enumerate(meths):
        r=[g for g in grid if g['inject']==inj and g['endpoint']==c and g['method']==mk and g['delta']=='2'][0]
        yy=y-j*0.25
        b.plot([float(r['ci90_lo']),float(r['ci90_hi'])],[yy,yy],color=col,lw=1.0)
        b.plot(float(r['mean_diff']),yy,mkr,color=col,ms=3.2,label=ml if y==0 else None)
    yt.append(y-0.25);yl.append(lab);y-=1
b.set_yticks(yt);b.set_yticklabels(yl);b.set_xlim(-3.6,3.6)
b.set_xlabel('Method minus Gold (pp), mean and 90% CI');b.set_xticks([-3,-2,-1.5,-1,0,1,1.5,2,3]);b.set_xticklabels(['-3','-2','','-1','0','1','','2','3'])
b.set_title('(b)',loc='left',fontsize=8,pad=3);b.tick_params(axis='y',right=False,length=0)
b.legend(loc='upper center',bbox_to_anchor=(0.5,-0.24),ncol=3,fontsize=7,frameon=False,handlelength=1.2)
fig.tight_layout(w_pad=1.2);fig.savefig(HERE/'out'/'fig_confirm.pdf')
