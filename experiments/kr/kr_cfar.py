"""CA-CFAR on the K-Radar power RA map (training 9x9, guard 3x3, threshold factor tuned on val among {2,3,4,6}); CPU."""
import numpy as np, sys, os, json
from scipy.ndimage import uniform_filter
sys.path.insert(0,".."); from eval_points import score; from kr_common import *
def cfar(a,f,tr=9,gd=3):
    s_tr=uniform_filter(a,size=tr,mode="nearest")*tr*tr; s_gd=uniform_filter(a,size=gd,mode="nearest")*gd*gd; noise=(s_tr-s_gd)/(tr*tr-gd*gd); return a>f*noise
def run(pp,f):
    out={}
    for fid,r,l in pp:
        a=ra(r); P=polar(cfar(a,f).astype(float),0.5,256,128)
        if len(P)<2: P=topn(a,50,256,128)
        out[fid]=(score(P,gt(l)),P)
    return out
val=pairs("val")[::max(1,len(pairs("val"))//40)][:40]; best=None
for f in (2,3,4,6):
    v=np.mean([s["CD"] for s,_ in run(val,f).values()]); print("VAL f=%g CD %.3f"%(f,v),flush=True)
    if best is None or v<best[0]: best=(v,f)
d=run(pairs("test"),best[1]); os.makedirs("points_cfar",exist_ok=True)
for fid,(s,P) in d.items(): np.concatenate([P,np.zeros((len(P),1))],1).astype(np.float32).tofile("points_cfar/%s.bin"%fid)
m={k:float(np.mean([s[k] for s,_ in d.values()])) for k in ["FID","CD","UCD","MHD","pts"]}
print("TEST CFAR (f=%g): CD %.3f UCD %.3f FID %.1f pts %.0f n=%d"%(best[1],m["CD"],m["UCD"],m["FID"],m["pts"],len(d))); json.dump(dict(m,f=best[1]),open("kr_cfar_test.json","w"))
