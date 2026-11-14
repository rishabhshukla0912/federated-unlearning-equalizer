import csv,json,glob,math,pathlib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]/'results'/'protocol_v2'
from scipy import stats
rows=[]
for f in sorted(glob.glob(str(ROOT/'runs'/'confirm_*'/'summary.csv'))):
    rows+= [r for r in csv.DictReader(open(f)) if r['split']=='test']
def get(inj,method,col):
    d={int(r['seed']):float(r[col]) for r in rows if r['inject']==inj and r['method']==method and r[col]!=''}
    return d
METHODS=['FedEraser','Grad-ascent','Retain-finetune']
ENDP={'canary':['canary_asr','retain_acc','forget_acc'],'iq':['forget_acc','retain_acc']}
out=[];lines=[]
def paired(d,g,delta):
    s=sorted(set(d)&set(g)); x=np.array([d[k]-g[k] for k in s]); n=len(x); m=x.mean(); sd=x.std(ddof=1); se=sd/math.sqrt(n); df=n-1
    if se==0: p=0.0 if abs(m)<delta else 1.0
    else: p=max(1-stats.t.cdf((m+delta)/se,df), stats.t.cdf((m-delta)/se,df))
    t=stats.t.ppf(0.95,df); return n,m,sd,m-t*se,m+t*se,p
def welch(a,b,delta):
    a=np.array(list(a.values()));b=np.array(list(b.values())); m=a.mean()-b.mean()
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b); se=math.sqrt(va+vb)
    if se==0: return m,(0.0 if abs(m)<delta else 1.0)
    df=(va+vb)**2/((va**2/(len(a)-1)) + (vb**2/(len(b)-1)) ) if va+vb>0 else 1
    p=max(1-stats.t.cdf((m+delta)/se,df), stats.t.cdf((m-delta)/se,df)); return m,p
for inj,eps in ENDP.items():
    for col in eps:
        g=get(inj,'FedAvg-retained (gold)',col)
        for meth in METHODS:
            d=get(inj,meth,col)
            for delta in [1,1.5,2,3]:
                n,m,sd,lo,hi,p=paired(d,g,delta); wm,wp=welch(d,g,delta)
                out.append(dict(inject=inj,endpoint=col,method=meth,delta=delta,n=n,mean_diff=round(m,3),sd=round(sd,3),ci90_lo=round(lo,3),ci90_hi=round(hi,3),p_tost_paired=round(p,4),equiv_paired=p<0.05,p_tost_welch=round(wp,4),equiv_welch=wp<0.05))
with open(pathlib.Path(__file__).resolve().parent/'out_tost_margin_grid.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
# per-seed gates/guards/runtimes
ps=[]
for f in sorted(glob.glob(str(ROOT/'runs'/'confirm_*'/'run_meta.json'))):
    d=json.load(open(f)); a=d['ascent_info']
    ps.append(dict(tag=d['tag'],seed=d['seed'],task_gate=d['task_gate_ok'],canary_gate_val=d['canary_gate_ok_val'],all_asr_val=d['all_asr_val'],guard_stop=a['stop'],violation_step=a.get('violation_step'),returned_step=a.get('returned_step'),pipeline_min=round(d['pipeline_s']/60,1)))
with open(pathlib.Path(__file__).resolve().parent/'out_per_seed.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(ps[0]));w.writeheader();w.writerows(ps)
for r in ps: print(r)
for inj in ['canary','iq']:
  for col in ['canary_asr','retain_acc','forget_acc']:
    for meth in ['FedAvg-all','FedAvg-retained (gold)']+METHODS:
        d=get(inj,meth,col)
        if d: v=np.array(list(d.values())); print(inj,col,meth,round(v.mean(),2),round(v.std(ddof=1),2))
for r in out:
    if r['delta']==2: print(r)
