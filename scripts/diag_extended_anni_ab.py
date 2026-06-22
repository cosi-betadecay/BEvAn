"""Prototype the EXTENDED annihilation_angle (two-photon OR single-photon+positron)
and A/B it against the current back-to-back score, before porting to physics_factors.

current : min over vertices of back-to-back cosine + energy-deficit penalty (two arms,
          arm-energy gate) -- mirrors physics_factors._per_subset_back_to_back.
extended: current, OR a single-photon+positron vertex -- a 511-consistent subset plus a
          co-located extra (positron) deposit next to it. Fires (low score) on the ~71%
          single-photon annihilations the current score misses (returns NaN/inf for).

A/B: delta_E + current_anni  vs  delta_E + extended_anni. Noised parser. No MEGAlib.
"""
import itertools
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "betadecay-analysis"))
from modeling.matrix_calculations import build_density_matrix_1d, lookup_density_values_1d
from utils.sim_text_reader import event_is_bdecay, iter_sim_events

SIM=os.path.join(os.path.dirname(__file__),"..","data","Activation.sim")
REF=511.0; TOL=3.0*(2.25/2.355); SIGMA=2.25/2.355; RNG=random.Random(0)
FLOOR=511.0-3.0*(2.25/2.355); TARGET=1022.0; SIG_E=250.0; LAM=0.3
D_CO=1.5; E_POS_MIN=60.0   # positron: co-located (<1.5cm), >60 keV deposit

def dist(a,b): return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5
def best_sub(E):
    n=len(E); best,idx=float("inf"),()
    for r in range(1,min(n+1,8)):
        for c in itertools.combinations(range(n),r):
            d=abs(sum(E[i] for i in c)-REF)
            if d<best: best,idx=d,c
    return best,set(idx)

def back_to_back(E,P):
    n=len(E)
    if n<3: return float("inf")
    total=sum(E); best=float("inf")
    for v in range(n):
        others=[k for k in range(n) if k!=v]
        dirs=[]
        for k in others:
            d=(P[k][0]-P[v][0],P[k][1]-P[v][1],P[k][2]-P[v][2]); nn=(d[0]**2+d[1]**2+d[2]**2)**0.5
            if nn>1e-9: dirs.append((d[0]/nn,d[1]/nn,d[2]/nn))
        mc=1.0
        for a in range(len(dirs)):
            for b in range(a+1,len(dirs)): mc=min(mc,sum(dirs[a][t]*dirs[b][t] for t in range(3)))
        e_arms=total-E[v]
        if e_arms>FLOOR:
            deficit=max(0.0,TARGET-e_arms); et=LAM*(1-math.exp(-deficit**2/(2*SIG_E**2)))
            best=min(best,mc+et)
    return best

def extended(E,P):
    bb=back_to_back(E,P)
    dE,S=best_sub(E)
    sp=float("inf")
    if dE<TOL and len(S)>=1:
        Sl=list(S)
        for k in range(len(E)):
            if k in S: continue
            dmin=min(dist(P[k],P[s]) for s in Sl)
            if dmin<D_CO and E[k]>E_POS_MIN:
                sp=min(sp,-1.0+0.2*(dmin/D_CO))   # annihilation-like, comparable scale
    return min(bb,sp)

rows=[]
for ev in iter_sim_events(SIM):
    if not ev.hits: continue
    E=[h.energy+RNG.gauss(0,SIGMA) for h in ev.hits]; P=[h.position for h in ev.hits]
    nh=len(ev.hits); dE,_=best_sub(E)
    cur=back_to_back(E,P); ext=extended(E,P)
    rows.append({"y":1 if event_is_bdecay(ev,REF,TOL) else 0,"bucket":1 if nh==1 else (2 if nh==2 else 3),
                 "dE":dE,"cur":cur if math.isfinite(cur) else float("nan"),
                 "ext":ext if math.isfinite(ext) else float("nan")})

# coverage: fraction of signal events with a finite anni value
sig=[r for r in rows if r["y"]==1]
cov_cur=sum(1 for r in sig if r["dE"]==r["dE"] and r["cur"]==r["cur"])/len(sig)
cov_ext=sum(1 for r in sig if r["dE"]==r["dE"] and r["ext"]==r["ext"])/len(sig)
print(f"signal coverage (finite anni): current={cov_cur:.3f}  extended={cov_ext:.3f}")

g=torch.Generator().manual_seed(42); idx=torch.randperm(len(rows),generator=g).tolist()
k=int(0.8*len(rows)); tr=[rows[i] for i in idx[:k]]; ev_=[rows[i] for i in idx[k:]]
def _t(rs,key): return torch.tensor([r[key] for r in rs],dtype=torch.float32)
def fin(rs,key): return [r for r in rs if r[key]==r[key]]
def f1_for(annikey):
    tp=fp=fn=tn=0
    for b in (1,2,3):
        trb_=[r for r in tr if r["bucket"]==b]; te=[r for r in ev_ if r["bucket"]==b]
        trs=[r for r in trb_ if r["y"]==1]; trg=[r for r in trb_ if r["y"]==0]
        if not trs or not trg or not te:
            fn+=sum(r["y"] for r in te); tn+=sum(1-r["y"] for r in te); continue
        prior=len(trg)/len(trs)
        _,xb=build_density_matrix_1d(_t(trb_,"dE"),n_bins_x=25,spacing_x="log",log_x_floor=1e-3)
        sb,_=build_density_matrix_1d(_t(trs,"dE"),x_bins=xb,smoothing=0.5)
        bb,_=build_density_matrix_1d(_t(trg,"dE"),x_bins=xb,smoothing=0.5)
        lr=(lookup_density_values_1d(_t(te,"dE"),sb,xb)+1e-8)/(lookup_density_values_1d(_t(te,"dE"),bb,xb)+1e-8)
        if annikey and b>=2:
            trs_a=fin(trs,annikey); trg_a=fin(trg,annikey)
            if trs_a and trg_a:
                lo=min(r[annikey] for r in trb_ if r[annikey]==r[annikey])
                xb2=torch.linspace(lo-0.01,1.01,16)
                sa,_=build_density_matrix_1d(_t(trs_a,annikey),x_bins=xb2,smoothing=0.5)
                ga,_=build_density_matrix_1d(_t(trg_a,annikey),x_bins=xb2,smoothing=0.5)
                # neutralize (LR factor=1) where anni is NaN
                vals=_t(te,annikey); finite=vals==vals
                fac=torch.ones(len(te))
                if finite.any():
                    pa=lookup_density_values_1d(vals[finite],sa,xb2)+1e-8
                    pg=lookup_density_values_1d(vals[finite],ga,xb2)+1e-8
                    fac[finite]=pa/pg
                lr=lr*fac
        pred=lr>=prior; yv=torch.tensor([r["y"] for r in te])
        tp+=int(((yv==1)&pred).sum()); fp+=int(((yv==0)&pred).sum())
        fn+=int(((yv==1)&~pred).sum()); tn+=int(((yv==0)&~pred).sum())
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    return (2*prec*rec/(prec+rec) if prec+rec else 0,tp,fp,fn,tn)

print(f"\nevents {len(rows)} (noised)")
for key,d in [(None,"delta_E only"),("cur","delta_E + current anni"),("ext","delta_E + EXTENDED anni")]:
    f1,tp,fp,fn,tn=f1_for(key); print(f"  {d:28s}: F1={f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
