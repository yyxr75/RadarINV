"""Table IV on the 46-frame K-Radar subset (every 8th test frame): all methods, same GT and script."""
import numpy as np, glob, os, sys, json
from scipy.spatial import cKDTree
sys.path.insert(0,".."); from eval_points import score; from kr_common import *
sub=open("subset_ids.txt").read().split(); P_={fid:(r,l) for fid,r,l in pairs("test")}
def ror(P,r,k):
    if r==0 or len(P)<k+2: return P
    n=cKDTree(P).query_ball_point(P,r,return_length=True)-1; Q=P[n>=k]; return Q if len(Q)>=2 else P
def ours(fid,ex,r,k):
    m=np.squeeze(np.load("ours_final/%s/%s_decoded.npy"%(fid,fid))); m=m.mean(0) if m.ndim==3 else m; H,W=m.shape
    P=topn(m,int(ex[3:]),H,W); return ror(P,r,k)
def ld(p): a=np.fromfile(p,np.float32); return a.reshape(-1,3)[:,:2]
rows={"CFAR":lambda f: ld("points_cfar/%s.bin"%f),"L1 Reg":lambda f: ld("points_l1/%s.bin"%f),"L2 Reg":lambda f: ld("points_l2/%s.bin"%f),
      "Radar-Diffusion":lambda f: ld("../radardiff_runs/points_kradar_db/%s.bin"%f),"Ours (val rule top8000)":lambda f: ours(f,"top8000",0,0),
      "Ours top5500":lambda f: ours(f,"top5500",0,0),"Ours top8000":lambda f: ours(f,"top8000",0,0),"Ours top3000":lambda f: ours(f,"top3000",0,0)}
res={}; per={}
for nm,fn in rows.items():
    s=[]
    for f in sub:
        P=fn(f); G=gt(P_[f][1]); r=score(P,G); s.append(r); per.setdefault(nm,[]).append(r["CD"])
    m={k:float(np.mean([x[k] for x in s])) for k in ["FID","CD","UCD","MHD","pts"]}; res[nm]=m
    print("%-40s FID %7.1f CD %6.3f UCD %6.3f MHD %6.3f pts %6.0f"%(nm,m["FID"],m["CD"],m["UCD"],m["MHD"],m["pts"]))
o=np.array(per["Ours (val rule top8000)"])
for b in ["L1 Reg","L2 Reg","CFAR","Radar-Diffusion"]:
    d=o-np.array(per[b]); rng=np.random.default_rng(0); bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)]
    print("Ours - %-16s %+.3f CI [%+.3f, %+.3f] better on %.0f%%"%(b,d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d<0).mean()))
json.dump(res,open("kr_table46.json","w"),indent=1)
