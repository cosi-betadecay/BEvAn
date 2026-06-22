"""Decisive build-or-not test: does a REALIZABLE annihilation-presence score
(trained on has_ANNI, train split only) help F1 as a model factor, or dilute
like total_E did?

A: delta_E only.  E: delta_E + realizable anni-score (MLP P(has_ANNI), held-out).
D: delta_E + has_ANNI oracle (upper bound, for reference).
Noised parser, per-bucket histogram-LR. No MEGAlib.
"""
import itertools
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from modeling.matrix_calculations import build_density_matrix_1d, lookup_density_values_1d
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM = os.path.join(os.path.dirname(__file__), "..", "data", "Activation.sim")
REF = 511.0; TOL = 3.0 * (2.25/2.355); SIGMA = 2.25/2.355; EDGE = 511.0*2/3; RNG = random.Random(0)

def best_sub(E):
    n=len(E); best,idx=float("inf"),()
    for r in range(1,min(n+1,8)):
        for c in itertools.combinations(range(n),r):
            d=abs(sum(E[i] for i in c)-REF)
            if d<best: best,idx=d,c
    return best,set(idx)

def feats(E,P,sub):
    n=len(E); tot=sum(E); out=[E[i] for i in range(n) if i not in sub]
    cx=sum(p[0] for p in P)/n; cy=sum(p[1] for p in P)/n; cz=sum(p[2] for p in P)/n
    rms=(sum((p[0]-cx)**2+(p[1]-cy)**2+(p[2]-cz)**2 for p in P)/n)**0.5
    return [tot,float(n),max(E),min(E),sum(out),max(out) if out else 0.0,
            float(len(out)),sum(1 for e in E if e>EDGE),rms]

rows=[]
for ev in iter_sim_events(SIM):
    if not ev.hits: continue
    E=[h.energy+RNG.gauss(0,SIGMA) for h in ev.hits]; P=[h.position for h in ev.hits]
    dE,sub=best_sub(E); nh=len(ev.hits)
    rows.append({"y":1 if event_is_bdecay(ev,REF,TOL) else 0,
                 "anni":1.0 if any(ia.process=="ANNI" for ia in ev.ias) else 0.0,
                 "bucket":1 if nh==1 else (2 if nh==2 else 3),"dE":dE,
                 "f":feats(E,P,sub) if nh>=2 else [0.0]*9})

g=torch.Generator().manual_seed(42); idx=torch.randperm(len(rows),generator=g).tolist()
k=int(0.8*len(rows)); tr_rows=[rows[i] for i in idx[:k]]; ev_rows=[rows[i] for i in idx[k:]]

# train realizable anni-score on TRAIN only (multi-hit), predict for all
trm=[r for r in tr_rows if r["bucket"]>=2]
Xtr=torch.tensor([r["f"] for r in trm],dtype=torch.float32)
mu,sd=Xtr.mean(0),Xtr.std(0).clamp_min(1e-6); Xtr=(Xtr-mu)/sd
ytr=torch.tensor([r["anni"] for r in trm],dtype=torch.float32)
net=torch.nn.Sequential(torch.nn.Linear(9,24),torch.nn.ReLU(),torch.nn.Linear(24,1))
opt=torch.optim.Adam(net.parameters(),lr=0.01,weight_decay=1e-4)
for _ in range(400):
    opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr).squeeze(1),ytr); loss.backward(); opt.step()
def score(r):
    if r["bucket"]<2: return 0.0
    x=(torch.tensor(r["f"],dtype=torch.float32)-mu)/sd
    with torch.no_grad(): return float(torch.sigmoid(net(x.unsqueeze(0)).squeeze()))
for r in rows: r["score"]=score(r)

def _t(rs,key): return torch.tensor([r[key] for r in rs],dtype=torch.float32)
def f1_for(mode):
    tp=fp=fn=tn=0
    for b in (1,2,3):
        tr=[r for r in tr_rows if r["bucket"]==b]; te=[r for r in ev_rows if r["bucket"]==b]
        trs=[r for r in tr if r["y"]==1]; trb=[r for r in tr if r["y"]==0]
        if not trs or not trb or not te:
            fn+=sum(r["y"] for r in te); tn+=sum(1-r["y"] for r in te); continue
        prior=len(trb)/len(trs)
        _,xb=build_density_matrix_1d(_t(tr,"dE"),n_bins_x=25,spacing_x="log",log_x_floor=1e-3)
        sb,_=build_density_matrix_1d(_t(trs,"dE"),x_bins=xb,smoothing=0.5)
        bb,_=build_density_matrix_1d(_t(trb,"dE"),x_bins=xb,smoothing=0.5)
        lr=(lookup_density_values_1d(_t(te,"dE"),sb,xb)+1e-8)/(lookup_density_values_1d(_t(te,"dE"),bb,xb)+1e-8)
        key={"E":"score","D":"anni"}.get(mode)
        if key and b>=2:
            xb2=torch.linspace(0,1,21) if key=="score" else torch.tensor([-0.5,0.5,1.5])
            ss,_=build_density_matrix_1d(_t(trs,key),x_bins=xb2,smoothing=0.5)
            bs,_=build_density_matrix_1d(_t(trb,key),x_bins=xb2,smoothing=0.5)
            lr=lr*(lookup_density_values_1d(_t(te,key),ss,xb2)+1e-8)/(lookup_density_values_1d(_t(te,key),bs,xb2)+1e-8)
        pred=lr>=prior; yv=torch.tensor([r["y"] for r in te])
        tp+=int(((yv==1)&pred).sum()); fp+=int(((yv==0)&pred).sum())
        fn+=int(((yv==1)&~pred).sum()); tn+=int(((yv==0)&~pred).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    return (2*prec*rec/(prec+rec) if prec+rec else 0,tp,fp,fn,tn)

print(f"events {len(rows)} (noised)")
for m,d in [("A","delta_E only"),("E","delta_E + realizable anni-score"),("D","delta_E + has_ANNI ORACLE")]:
    f1,tp,fp,fn,tn=f1_for(m); print(f"  [{m}] {d:34s}: F1={f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
