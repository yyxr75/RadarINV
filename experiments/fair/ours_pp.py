"""Ours post-processing chosen on 30 val frames: extraction (top-B, Otsu) x radius-outlier removal (r,k); applied once to the 424 test frames."""
import numpy as np, glob, os, sys, json
from skimage.filters import threshold_otsu
from scipy.spatial import cKDTree
sys.path.insert(0,".."); from eval_points import score
D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; AZ=np.load("../../CalibrationTable.npy",allow_pickle=True).item()["Azimuth_table"]
def load(p):
    m=np.squeeze(np.load(p)); return m.mean(0) if m.ndim==3 else m
def pts(m,ex):
    H,W=m.shape; azn=np.deg2rad(np.linspace(AZ[0],AZ[-1],W)); rr=np.linspace(0,H,H)/H*103
    th=threshold_otsu(m) if ex=="otsu" else np.sort(m.ravel())[-int(ex[3:])]; iy,ix=np.where(m>=th); return np.stack([rr[iy]*np.cos(azn[ix]),rr[iy]*np.sin(azn[ix])],1)
def ror(P,r,k):
    if r==0: return P
    n=cKDTree(P).query_ball_point(P,r,return_length=True)-1; Q=P[n>=k]; return Q if len(Q)>=2 else P
EXS=["otsu","top3000","top5500","top8000"]; RK=[(0,0),(0.5,2),(0.5,4),(1.0,4),(1.0,8),(2.0,8),(2.0,16)]
val=sorted(glob.glob("ours_val30/*/*_decoded.npy")); grid={}
for p in val:
    f=os.path.basename(p)[:6]; m=load(p); G=np.fromfile(D+"lidars_no_ground/%s.bin"%f,np.float32).reshape(-1,3)[:,:2]
    for ex in EXS:
        P0=pts(m,ex)
        for r,k in RK: grid.setdefault((ex,r,k),[]).append(score(ror(P0,r,k),G)["CD"])
for key in sorted(grid,key=lambda t:np.mean(grid[t]))[:6]: print("VAL",key,"CD %.3f"%np.mean(grid[key]))
best=min(grid,key=lambda t:np.mean(grid[t])); print("CHOSEN",best,"val CD %.3f (no-post top5500 %.3f)"%(np.mean(grid[best]),np.mean(grid[("top5500",0,0)])),flush=True)
os.makedirs("points_ours_pp",exist_ok=True); res=[]
for p in sorted(glob.glob("../full/out*/*/*_decoded.npy")):
    f=os.path.basename(p)[:6]; P=ror(pts(load(p),best[0]),best[1],best[2]); np.concatenate([P,np.zeros((len(P),1))],1).astype(np.float32).tofile("points_ours_pp/%s.bin"%f)
    res.append(score(P,np.fromfile(D+"lidars_no_ground/%s.bin"%f,np.float32).reshape(-1,3)[:,:2]))
m={k:float(np.mean([s[k] for s in res])) for k in ["FID","CD","UCD","HD","UMHD","pts"]}; print("TEST Ours+post: CD %.3f UCD %.3f FID %.1f HD %.1f UMHD %.1f pts %.0f n=%d"%(m["CD"],m["UCD"],m["FID"],m["HD"],m["UMHD"],m["pts"],len(res)))
json.dump(dict(m,choice=list(best)),open("ours_pp_test.json","w"))
