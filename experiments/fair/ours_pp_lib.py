"""Shared extraction helpers of ours_pp.py."""
import numpy as np
from skimage.filters import threshold_otsu
from scipy.spatial import cKDTree
AZ=np.load("../../CalibrationTable.npy",allow_pickle=True).item()["Azimuth_table"]
def load(p):
    m=np.squeeze(np.load(p)); return m.mean(0) if m.ndim==3 else m
def pts(m,ex):
    H,W=m.shape; azn=np.deg2rad(np.linspace(AZ[0],AZ[-1],W)); rr=np.linspace(0,H,H)/H*103
    th=threshold_otsu(m) if ex=="otsu" else np.sort(m.ravel())[-int(ex[3:])]; iy,ix=np.where(m>=th); return np.stack([rr[iy]*np.cos(azn[ix]),rr[iy]*np.sin(azn[ix])],1)
def ror(P,r,k):
    if r==0: return P
    n=cKDTree(P).query_ball_point(P,r,return_length=True)-1; Q=P[n>=k]; return Q if len(Q)>=2 else P
