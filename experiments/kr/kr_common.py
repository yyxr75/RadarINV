"""K-Radar helpers: test/val pairs, RA (elevation-mean dB cube -> power -> min-max), GT points (paper rule), geometry 0..118.037 m, +-53 deg."""
import numpy as np, os
K_=os.environ["KRADAR_ROOT"].rstrip("/") + "/"; R=118.037; AZD=53.0
def pairs(split):
    out=[]
    for line in open(K_+("testset.txt" if split=="test" else "valset.txt")):
        r,l=line.split(); r=K_+r.rsplit("/data/",1)[1]; l=K_+l.rsplit("/data/",1)[1]; out.append((r.split("/")[-3]+"_"+os.path.basename(r)[5:-4],r,l))
    return out
def ra(r):
    a=np.fromfile(r,np.float32).reshape(256,32,128).mean(1); a=10**(a/10); return (a-a.min())/(a.max()-a.min()+1e-12)
def gt(l):
    mm=np.fromfile(l,np.float32).reshape(256,32,128).mean(1); m=mm>mm.mean()+0.01; return polar(m.astype(float),0.5,256,128)
def polar(m,th,H,W):
    iy,ix=np.where(m>=th); rr=(iy+0.5)/H*R; az=np.deg2rad(-AZD+(ix+0.5)/W*2*AZD); return np.stack([rr*np.cos(az),rr*np.sin(az)],1)
def topn(m,n,H,W): return polar(m,np.sort(m.ravel())[-min(n,m.size)],H,W)
