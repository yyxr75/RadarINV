"""Fig. 3: forward operator A(x) of Section III-B on RADIal LiDAR occupancy maps, for ideal half-wavelength ULAs of N elements.
Same steps as RadarImagingOperator.cond_fn: mask the spatial signal cube, average over the angle axis, W-point zero-padded
FFT along the antenna axis, shift, magnitude. Display: on the BEV plane (range 0..103 m, azimuth -75..75 deg); A(x) in dB relative to
each panel's peak, one shared color scale."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; FR=['003641','003712','003887','013861','014012']; NS=[4,12,86,192]; H,W=512,768
th=np.deg2rad(np.arange(W)*150.0/W-75.0)                     # same azimuth grid as the operator
def A(x,N):
    S=np.exp(1j*np.arange(N)[:,None]*np.pi*np.sin(th)[None,:])  # d = lambda/2  ->  2*pi*d/lambda = pi
    M=x@S.T/W                                                   # H x N: masked cube averaged over the angle axis
    F=np.fft.fft(M,n=W,axis=1); F=np.roll(F,W//2,axis=1)        # zero-padded angle FFT, shifted
    return np.abs(F)
VMIN=-25.0
RE,TE=np.meshgrid(np.arange(H+1)/H*103.0,np.deg2rad(np.arange(W+1)*150.0/W-75.0),indexing="ij")   # cell edges of the same range-azimuth grid
def show(a,img,vmin,vmax):   # drawn on the BEV plane: range 0..103 m, azimuth -75..75 deg
    a.pcolormesh(RE*np.sin(TE),RE*np.cos(TE),img,cmap="viridis",vmin=vmin,vmax=vmax,shading="flat",rasterized=True)
    L=103*np.sin(np.deg2rad(75)); a.set_xlim(-L-2,L+2); a.set_ylim(-2,105); a.set_aspect("auto")
fig,ax=plt.subplots(1+len(NS),len(FR),figsize=(1.75*len(FR)+0.35,1.3*(1+len(NS))))
for c,f in enumerate(FR):
    x=np.fromfile(D+"lidars_mask/%s.bin"%f).reshape(H,W); x=(x>0.5).astype(float)
    show(ax[0,c],x,0,1)
    for r,N in enumerate(NS):
        y=A(x,N); db=20*np.log10(np.maximum(y/y.max(),1e-6))
        show(ax[r+1,c],np.clip(db,VMIN,0),VMIN,0)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.5); s.set_edgecolor("#b3b3b3")
labels=[r"$\mathbf{x}$"]+[r"$N=%d$"%N for N in NS]
for r,l in enumerate(labels): ax[r,0].set_ylabel(l,fontsize=9,rotation=90,labelpad=3)
plt.subplots_adjust(left=0.06,right=0.995,top=0.995,bottom=0.005,wspace=0.04,hspace=0.04)
plt.savefig("fig3_bev.pdf",dpi=220); plt.savefig("fig3_bev.png",dpi=90); print("saved")
