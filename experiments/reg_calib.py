"""L1/L2 regularised inversion with the calibrated operator of the chosen virtual-channel subset.
usage: reg_calib.py <frames.txt> <y_dir> <ntx> <nrx> <out.json>
A_n(x)[h,:] = | x[h,:] @ K_n^T |,  K_n = (Cal768[:,idx]*w[idx]) @ Cal768[:,idx]^H * 192/n   (same as the sampler's operator)
lambda chosen per setting from a grid, best CD kept (favours the baseline). Points: Otsu threshold on the recovered map.
"""
import numpy as np, torch, os, sys, json
from skimage.filters import threshold_otsu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from eval_points import score
D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; dev=torch.device("cuda:0"); H,W=512,768
C=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CalibrationTable.npy"),allow_pickle=True).item()
AZ=C['Azimuth_table']; azn=np.linspace(AZ[0],AZ[-1],W)
def K_n(ntx,nrx):
    S=C['Signal'][...,5]; w=C['H'][0].ravel(); idx=np.array([tx*16+rx for tx in range(ntx) for rx in range(nrx)])
    S768=np.stack([np.interp(azn,AZ,S[:,a].real)+1j*np.interp(azn,AZ,S[:,a].imag) for a in range(192)],1)
    K=(S768[:,idx]*w[idx][None,:])@S768[:,idx].conj().T*(192.0/len(idx))
    return torch.from_numpy((K/2450.0).astype(np.complex64)).to(dev)
def A(x,K): return torch.abs(torch.matmul(x.to(K.dtype),K.transpose(0,1)))
def solve(y,K,reg,lam,iters=300,lr=0.05):
    g=torch.Generator(device='cpu').manual_seed(0); x=(0.01*torch.rand(y.shape,generator=g)).to(dev).requires_grad_(True)
    opt=torch.optim.Adam([x],lr=lr)
    for _ in range(iters):
        opt.zero_grad(); r=A(x,K)-y; loss=(r*r).sum()+lam*(x.abs().sum() if reg=='l1' else (x*x).sum()); loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(min=0)
    return x.detach().cpu().numpy()
def to_xy(m,th=None):
    th=threshold_otsu(m) if th is None else th
    iy,ix=np.where(m>th); rr=np.linspace(0,H,H)/H*103; a=np.deg2rad(azn[ix]); return np.stack([rr[iy]*np.cos(a),rr[iy]*np.sin(a)],1)
def extractions(m):   # Otsu plus fixed point budgets; the best CD is kept, which favours the baseline
    v=np.sort(m.ravel()); return [("otsu",None)]+[("top%d"%n, v[-n]) for n in (400,800,1500,3000)]
frames=open(sys.argv[1]).read().split(); ydir=sys.argv[2]; ntx,nrx=int(sys.argv[3]),int(sys.argv[4]); K=K_n(ntx,nrx)
G={f:np.fromfile(D+"lidars_no_ground/%s.bin"%f,np.float32).reshape(-1,3)[:,:2] for f in frames}
res={}
for reg in ["l1","l2"]:
    best=None
    for lam in ([0.002,0.01,0.05,0.2] if reg=="l1" else [0.01,0.05,0.2,1.0]):
        per={}
        for k in range(0,len(frames),8):
            ch=frames[k:k+8]; Y=torch.stack([torch.from_numpy(np.fromfile(ydir+"/%s.bin"%f).reshape(H,W)).float() for f in ch]).to(dev)
            X=solve(Y,K,reg,lam)
            for f,x in zip(ch,X):
                for name,th in extractions(x):
                    P=to_xy(x,th)
                    if len(P)>=2: per.setdefault(name,{})[f]=score(P,G[f])
        for name,pp in per.items():
            s={k:float(np.mean([v[k] for v in pp.values()])) for k in ["FID","CD","UCD","HD","UMHD","pts"]}; s["lam"]=lam; s["extract"]=name; s["n"]=len(pp)
            print("%s ntx=%d nrx=%d lam=%g %-8s CD %.3f UCD %.3f pts %.0f n=%d"%(reg,ntx,nrx,lam,name,s["CD"],s["UCD"],s["pts"],s["n"]),flush=True)
            if best is None or s["CD"]<best["CD"]: best=s
    res[reg]=best
json.dump(res,open(sys.argv[5],"w"),indent=1); print("BEST",{k:(round(v["CD"],3),v["lam"],v["extract"]) for k,v in res.items()})
