"""Fig. 5 (K-Radar cross-dataset) = the submitted figure's seven panels per row, plus a Radar-Diffusion zero-shot panel."""
import numpy as np, os, glob, torch, torch.nn.functional as F, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
SP=os.path.dirname(os.path.abspath(__file__)); K=os.environ["KRADAR_ROOT"].rstrip("/") + "/" + "1/"; R=118.037; A=np.deg2rad(53)
ROWS=[(203,172),(248,217),(258,227),(272,241),(365,334),(322,291)]   # (radar cube, LiDAR frame) identified from the submitted figure
strips=sorted(glob.glob(SP+"/fig5_rows/row-*.png"))
titles=["Radar RA Input","CFAR","L1 Reg","L2 Reg","RadarHD","Diffradar","Radar-Diffusion","Ours"]
def box(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#b3b3b3")
PITCH=327.6; X0,X1=10,287   # Radar RA Input image spans columns 10..286 of each strip; the other panels repeat at this pitch
def rdb(c): m=np.fromfile(K+"radar_rea_cube/cube_%05d.bin"%c,np.float32).reshape(256,32,128).mean(1); return m-np.median(m)
VMAX=np.percentile(np.concatenate([rdb(c).ravel() for c,_ in ROWS]),99.9)
fig,ax=plt.subplots(len(ROWS),8,figsize=(1.75*8,1.6*len(ROWS)))
for r,((cube,lid),s) in enumerate(zip(ROWS,strips)):
    im=np.array(Image.open(s).convert("RGB")); nw=np.where((im[:,X0:X1].astype(int).sum(2)<720).any(1))[0]; Y0,Y1=nw.min(),nw.max()+1
    panels=[im[Y0:Y1,X0+int(round(i*PITCH)):X1+int(round(i*PITCH))] for i in range(7)]   # every panel cropped to the RA image's box
    h,w=panels[0].shape[:2]
    order=panels[:6]+[None]+[panels[6]]
    for c,(name,p) in enumerate(zip(titles,order)):
        a=ax[r,c]; box(a); a.set_box_aspect(h/w)
        xs=2*R*np.sin(A)/254; ys=R/223   # the submitted panels place the sector at columns 12..265 and rows 10..232 of the crop
        if c==0:   # radar input redrawn on the polar sector: elevation-mean dB, shown above the frame median, common window
            Rr,T=np.meshgrid(np.linspace(0,R,257),np.deg2rad(np.linspace(-53,53,129)),indexing="ij")
            a.pcolormesh(Rr*np.sin(T),Rr*np.cos(T),rdb(cube),cmap="viridis",vmin=0,vmax=VMAX,shading="flat",rasterized=True)
            a.set_xlim(-R*np.sin(A)-12*xs,R*np.sin(A)+(w-266)*xs); a.set_ylim(-(h-233)*ys,R+10*ys); a.set_aspect("auto")
        elif p is not None: a.imshow(p,aspect="auto")
        else:
            mm=np.fromfile(K+"lidar_rea_mask/os2-64_%05d.bin"%lid,np.float32).reshape(256,32,128).mean(1)
            mm=F.interpolate(torch.tensor(mm/mm.max())[None,None],(128,192),mode="bilinear").numpy()[0,0]; m=mm>mm.mean()+0.01   # the paper's K-Radar GT path
            iy,ix=np.where(m); rr=np.linspace(0,128,128)[iy]/128*R; th=np.deg2rad(np.linspace(-53,53,192)[ix])
            P=np.fromfile(SP+"/radardiff_runs/points_kradar_fig/1_%05d.bin"%cube,np.float32).reshape(-1,3)
            a.scatter(rr[::2]*np.sin(th[::2]),rr[::2]*np.cos(th[::2]),s=0.25,c="blue",lw=0,rasterized=True); a.scatter(P[:,1],P[:,0],s=0.6,c="red",lw=0,rasterized=True)
            a.set_xlim(-R*np.sin(A)-12*xs,R*np.sin(A)+(w-266)*xs); a.set_ylim(-(h-233)*ys,R+10*ys); a.set_aspect("auto")
plt.subplots_adjust(left=0.005,right=0.995,top=0.95,bottom=0.005,wspace=0.04,hspace=0.04)
for c,name in enumerate(titles):
    b=ax[0,c].get_position(); fig.text((b.x0+b.x1)/2,0.972,name,ha="center",va="center",fontsize=9,fontweight="bold")
out=SP+"/fig5_polar.pdf"; plt.savefig(out,dpi=220); plt.savefig(SP+"/fig5_polar.png",dpi=90); print("saved",out)
