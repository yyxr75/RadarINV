"""Fig. 4 (final): six test frames (RadarHD output < 25k points; 4 countryside, 2 highway), real-radar outputs of every method."""
import numpy as np, os, matplotlib;
from scipy.spatial import cKDTree
def ror(P,r=1.0,k=8):
    if len(P)<k+2: return P
    n=cKDTree(P).query_ball_point(P,r,return_length=True)-1; Q=P[n>=k]; return Q if len(Q)>=2 else P
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; FR=['003641','003712','003887','003946','013861','014012']
def ld(p,c=3): a=np.fromfile(p,np.float32); return a.reshape(-1,c)[:,:2]
cols=[("Camera",None,0),("Radar RA Input",None,0),("CFAR",D+"radars_pcd",4),("L1 Reg","points_l1",3),("L2 Reg","points_l2",3),("RadarHD",D+"radars_unet",3),
      ("Diffradar","../diffradar/test_diffradar/radar_points",3),("Radar-Diffusion","../radardiff_runs/points_final_test",3),("Ours","points_ours_pp",3)]
def box(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#b3b3b3")
def radb(f):   # stored maps are peak-normalized per frame, so show dB above the frame median (its noise floor) for a common background
    m=np.maximum(np.fromfile(D+"radars_ra_interp/%s.bin"%f).reshape(512,768),1e-12); return 10*np.log10(m/np.median(m))
AZ=np.load("../../CalibrationTable.npy",allow_pickle=True).item()["Azimuth_table"]
RE,TE=np.meshgrid(np.arange(513)/512*103.0,np.deg2rad(np.interp(np.arange(769)/768*(len(AZ)-1),np.arange(len(AZ)),AZ)),indexing="ij")   # cell edges: the paper's polar grid (rows range 0..103 m, columns on the azimuth table)
pool=np.concatenate([radb(f).ravel() for f in FR]); VMIN,VMAX=0.0,np.percentile(pool,99.9); print("dB window",VMIN,VMAX)
fig,ax=plt.subplots(len(FR),len(cols),figsize=(1.75*len(cols),1.75*len(FR)))
for r,f in enumerate(FR):
    G=ld(D+"lidars_no_ground/%s.bin"%f)
    for c,(nm,d,k) in enumerate(cols):
        a=ax[r,c]; box(a)
        if nm=="Camera": a.imshow(Image.open(D+"images/%s.png"%f),aspect="auto")   # scaled to the Radar RA Input panel size
        elif nm=="Radar RA Input": a.pcolormesh(RE*np.sin(TE),RE*np.cos(TE),radb(f),cmap="viridis",vmin=VMIN,vmax=VMAX,shading="flat",rasterized=True); a.set_xlim(-60,60); a.set_ylim(0,103); a.set_aspect("auto")   # drawn on the BEV plane like the point panels
        else:
            P=ld(os.path.join(d,f+".bin"),k); P=ror(P) if nm=="Ours" else P   # display-only outlier filter on our output (stated in the caption)
            a.scatter(G[::3,1],G[::3,0],s=0.2,c="blue",lw=0,rasterized=True); a.scatter(P[:,1],P[:,0],s=0.6,c="red",lw=0,rasterized=True); a.set_xlim(-60,60); a.set_ylim(0,103); a.set_aspect("auto")
plt.subplots_adjust(left=0.005,right=0.995,top=0.96,bottom=0.005,wspace=0.04,hspace=0.04)
for c,(nm,_,_) in enumerate(cols):
    b=ax[0,c].get_position(); fig.text((b.x0+b.x1)/2,0.975,nm,ha="center",va="center",fontsize=9,fontweight="bold")
out="fig4_cart.pdf"; plt.savefig(out,dpi=220); plt.savefig("fig4_cart.png",dpi=70); print("saved")
