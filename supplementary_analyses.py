#!/usr/bin/env python3
"""Supplementary Analyses — FAST (no LASSO, use fixed 3-Doppler features + GPU)"""
import os,pickle,warnings,time,numpy as np,pandas as pd
from scipy import stats; from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"]="32"
BASE=r"C:\Users\ZhangErHao\Desktop"
_z=[d for d in os.listdir(BASE) if d.startswith("dataset")]
BD=os.path.join(BASE,_z[0]); OUT=os.path.join(BASE,"重新验证后的论文数据")
PID,LABEL="patient_name","label"
XGB_KW=dict(n_jobs=8,tree_method="hist",device="cuda",verbosity=0)
SEEDS=[42,123,2024,7,999,314,271,1618,2048,4096]
NBINS=10; edges=np.linspace(0,1,NBINS+1); cents=(edges[:-1]+edges[1:])/2

# Load pre-built data (use existing pipeline's output)
# The 3 optimal Doppler features: VFA (flow_density), bf_firstorder_Energy, bf_glrlm_GrayLevelNonUniformity
DOPPLER_3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]

import re
def sid(r):
    m=re.search(r"(\d+)",str(r)); return int(m.group(1)) if m else None

# Load grayscale (1mm)
bus0=pd.read_csv(os.path.join(BD,"BUS_features","benign_0mm.csv"))
bus0_m=pd.read_csv(os.path.join(BD,"BUS_features","malignant_0mm.csv"))
bus0[LABEL],bus0_m[LABEL]=0,1; g0=pd.concat([bus0,bus0_m],ignore_index=True)
g0[PID]=g0[PID].apply(sid)
bus1=pd.read_csv(os.path.join(BD,"BUS_features","benign_1mm.csv"))
bus1_m=pd.read_csv(os.path.join(BD,"BUS_features","malignant_1mm.csv"))
bus1[LABEL],bus1_m[LABEL]=0,1; g1=pd.concat([bus1,bus1_m],ignore_index=True)
g1[PID]=g1[PID].apply(sid)

# Drop non-feature columns
def feat_cols(df):
    return [c for c in df.columns if c not in [PID,LABEL] and
            not any(p in c.lower() for p in ["patient_name","file_name","diagnostics_"])]

g0f=g0[feat_cols(g0)+[PID,LABEL]]; g1f=g1[feat_cols(g1)+[PID,LABEL]]
g1f=g1f.drop(columns=[c for c in g1f.columns if "shape" in c.lower() and c not in [PID,LABEL]],errors="ignore")
g1f=g1f.rename(columns={c:c+"_1mm" if c not in [PID,LABEL] else c for c in g1f.columns})

# Merge 0mm + 1mm
df=pd.merge(g0f,g1f,on=[PID,LABEL],how="inner")
gray_feats=[c for c in df.columns if c not in [PID,LABEL]]

# Load Doppler features
bf=pd.read_csv(os.path.join(BD,"flow_features","blood_flow_features_benign.csv"))
bf_m=pd.read_csv(os.path.join(BD,"flow_features","blood_flow_features_malignant.csv"))
bf[LABEL],bf_m[LABEL]=0,1; bf=pd.concat([bf,bf_m],ignore_index=True)
bf[PID]=bf["image_id"].apply(sid)
bf=bf[[PID,LABEL]+[c for c in bf.columns if c.startswith("bf_")]]

vfa=pd.read_csv(os.path.join(BD,"flow_features","benign_flow_density.csv"))
vfa_m=pd.read_csv(os.path.join(BD,"flow_features","malignant_flow_density.csv"))
vfa[LABEL],vfa_m[LABEL]=0,1; vfa=pd.concat([vfa,vfa_m],ignore_index=True)
vfa[PID]=vfa["case_id"].apply(sid); vfa["flow_density"]=vfa["flow_density"].astype(float)
vfa=vfa[[PID,"flow_density"]]

df_d=pd.merge(df,bf,on=[PID,LABEL],how="inner")
df_d=pd.merge(df_d,vfa,on=PID,how="inner")
all_feats=[c for c in df_d.columns if c not in [PID,LABEL,"flow_density"]]

X_gray=df[gray_feats].values; y=df[LABEL].values.ravel()
X_dopp=df_d[all_feats].values  # grayscale + all Doppler
y_d=df_d[LABEL].values.ravel()

# Also build a version with only 3 Doppler features
gray_only=[c for c in all_feats if not c.startswith("bf_") and c!="flow_density"]
dopp3=gray_only+DOPPLER_3
X_d3=df_d[[c for c in dopp3 if c in df_d.columns]].values

print(f"Gray: {X_gray.shape}  D3: {X_d3.shape}  D14: {X_dopp.shape}")

def compute_all():
    t0=time.time()
    nr,br_g,br_d,auc_g,auc_d=[],[],[],[],[]
    cal_g,cal_d,dca_g,dca_d=[],[],[],[]

    for idx,s in enumerate(SEEDS):
        tr,te=train_test_split(np.arange(len(y)),test_size=0.3,random_state=s,stratify=y)

        # Gray
        sg=StandardScaler()
        Xt=sg.fit_transform(X_gray[tr]); Xe=sg.transform(X_gray[te])
        cg=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=s,**XGB_KW)
        cg.fit(Xt,y[tr]); pg=cg.predict_proba(Xe)[:,1]

        # Doppler 3
        sd=StandardScaler()
        Xt=sd.fit_transform(X_d3[tr]); Xe=sd.transform(X_d3[te])
        cd=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=s,**XGB_KW)
        cd.fit(Xt,y_d[tr]); pd_=cd.predict_proba(Xe)[:,1]

        ag=roc_auc_score(y[te],pg); ad=roc_auc_score(y_d[te],pd_)
        auc_g.append(ag); auc_d.append(ad)
        br_g.append(brier_score_loss(y[te],pg))
        br_d.append(brier_score_loss(y_d[te],pd_))

        # NRI
        e_up=np.sum(pd_[y[te]==1]>pg[y[te]==1]); e_dn=np.sum(pd_[y[te]==1]<pg[y[te]==1])
        n_up=np.sum(pd_[y[te]==0]>pg[y[te]==0]); n_dn=np.sum(pd_[y[te]==0]<pg[y[te]==0])
        ne,nn=np.sum(y[te]==1),np.sum(y[te]==0)
        nri_v=(e_up-e_dn)/ne+(n_dn-n_up)/nn
        se=np.sqrt((e_up+e_dn)/ne**2+(n_up+n_dn)/nn**2) if ne>0 and nn>0 else np.nan
        z=nri_v/se if se>0 else np.nan
        p_nri=2*(1-stats.norm.cdf(abs(z))) if not np.isnan(z) else 1.0
        nr.append({"nri":nri_v,"event":(e_up-e_dn)/ne,"non_event":(n_dn-n_up)/nn,"p":p_nri})

        # Calibration
        cg_,cd_=np.full(NBINS,np.nan),np.full(NBINS,np.nan)
        for i in range(NBINS):
            m=(pg>=edges[i])&(pg<edges[i+1]);
            if i==NBINS-1: m=(pg>=edges[i])&(pg<=edges[i+1])
            if m.sum(): cg_[i]=y[te][m].mean()
            m=(pd_>=edges[i])&(pd_<edges[i+1])
            if i==NBINS-1: m=(pd_>=edges[i])&(pd_<=edges[i+1])
            if m.sum(): cd_[i]=y_d[te][m].mean()
        cal_g.append(cg_); cal_d.append(cd_)

        # DCA
        ths=np.arange(0.01,0.99,0.01); nbg,nbd=np.empty(len(ths)),np.empty(len(ths))
        for i,pt in enumerate(ths):
            b=(pg>=pt).astype(int); nbg[i]=np.sum((b==1)&(y[te]==1))/len(y[te])-np.sum((b==1)&(y[te]==0))/len(y[te])*pt/(1-pt)
            b=(pd_>=pt).astype(int); nbd[i]=np.sum((b==1)&(y_d[te]==1))/len(y_d[te])-np.sum((b==1)&(y_d[te]==0))/len(y_d[te])*pt/(1-pt)
        dca_g.append(nbg); dca_d.append(nbd)

        print(f"  Seed {s:4d} ({idx+1}/{len(SEEDS)}): AUC g={ag:.4f} d={ad:.4f} dAUC={ad-ag:+.4f} NRI={nri_v:+.4f}",flush=True)

    # Results
    nri_v=[n["nri"] for n in nr]
    _,pn=stats.ttest_1samp(nri_v,0)
    _,pb=stats.ttest_rel(br_g,br_d)
    _,pa=stats.ttest_rel(auc_g,auc_d)
    cgm=np.nanmean(cal_g,0); cdm=np.nanmean(cal_d,0)
    eg=np.nanmean(np.abs(cgm-cents)); ed=np.nanmean(np.abs(cdm-cents))
    dnb_g=np.mean(dca_g,0); dnb_d=np.mean(dca_d,0)
    impr=dnb_d-dnb_g; ben=ths[impr>0.001]

    print(f"\n=== RESULTS ===")
    print(f"NRI: {np.mean(nri_v):.4f}+-{np.std(nri_v):.4f} p={pn:.4f}")
    print(f"  Event={np.mean([n['event'] for n in nr]):.4f} Non-event={np.mean([n['non_event'] for n in nr]):.4f}")
    print(f"Brier: G={np.mean(br_g):.4f} D={np.mean(br_d):.4f} d={np.mean(br_d)-np.mean(br_g):+.4f} p={pb:.4f}")
    print(f"AUC: G={np.mean(auc_g):.4f} D={np.mean(auc_d):.4f} d={np.mean(auc_d)-np.mean(auc_g):+.4f} p={pa:.4f}")
    print(f"Cal Eavg: G={eg:.4f} D={ed:.4f}")
    print(f"DCA benefit: {ben[0]:.2f}-{ben[-1]:.2f}" if len(ben)>1 else "DCA: no benefit")

    # HP Sensitivity (3 configs only, 5 seeds)
    print(f"\nHP Sensitivity:")
    hp_r=[]
    for nm,d,lr,n in [("Default(d3,lr0.1)",3,0.1,100),("Deep(d5,lr0.1)",5,0.1,100),("MoreN(d3,n500)",3,0.1,500)]:
        ds=[]
        for sd in SEEDS[:5]:
            tr,te=train_test_split(np.arange(len(y)),test_size=0.3,random_state=sd,stratify=y)
            sg=StandardScaler(); Xt=sg.fit_transform(X_gray[tr]); Xe=sg.transform(X_gray[te])
            c=xgb.XGBClassifier(max_depth=d,learning_rate=lr,n_estimators=n,random_state=sd,**XGB_KW)
            c.fit(Xt,y[tr]); ag_=roc_auc_score(y[te],c.predict_proba(Xe)[:,1])
            sd2=StandardScaler(); Xt=sd2.fit_transform(X_d3[tr]); Xe=sd2.transform(X_d3[te])
            c_=xgb.XGBClassifier(max_depth=d,learning_rate=lr,n_estimators=n,random_state=sd,**XGB_KW)
            c_.fit(Xt,y_d[tr]); ad_=roc_auc_score(y_d[te],c_.predict_proba(Xe)[:,1])
            ds.append(ad_-ag_)
        _,p=stats.ttest_1samp(ds,0)
        hp_r.append({"cfg":nm,"dAUC":np.mean(ds),"p":p})
        print(f"  {nm:25s}: dAUC={np.mean(ds):+.4f} p={p:.4f}")

    # Save
    pkg={
        "nri":{"mean":float(np.mean(nri_v)),"std":float(np.std(nri_v)),"p":float(pn),
               "event":float(np.mean([n["event"] for n in nr])),
               "non_event":float(np.mean([n["non_event"] for n in nr]))},
        "brier":{"gray":float(np.mean(br_g)),"dopp":float(np.mean(br_d)),
                 "delta":float(np.mean(br_d)-np.mean(br_g)),"p":float(pb)},
        "auc":{"gray":float(np.mean(auc_g)),"dopp":float(np.mean(auc_d)),
               "delta":float(np.mean(auc_d)-np.mean(auc_g)),"p":float(pa)},
        "calibration":{"eavg_g":float(eg),"eavg_d":float(ed),
                        "centers":cents.tolist(),"gray_fracs":cgm.tolist(),"dopp_fracs":cdm.tolist()},
        "dca":{"thresholds":ths.tolist(),"nb_gray":dnb_g.tolist(),"nb_dopp":dnb_d.tolist(),
                "benefit_range":[float(ben[0]),float(ben[-1])] if len(ben)>1 else []},
        "hp_sensitivity":hp_r,
    }
    with open(os.path.join(OUT,"supplementary_analyses.pkl"),"wb") as f: pickle.dump(pkg,f)
    print(f"\n[OK] {time.time()-t0:.0f}s")

if __name__=="__main__":
    compute_all()
