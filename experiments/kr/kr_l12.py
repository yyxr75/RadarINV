"""K-Radar real-radar L1/L2 (+TV): ideal 84-antenna steering on the native 256x128 polar grid (+-53 deg), projected Adam.
Tune (reg weight, TV, extraction) on 40 val frames; run all test frames; save points."""
import numpy as np, torch, sys, os, json
sys.path.insert(0,".."); from eval_points import score; from kr_common import *
dev=torch.device("cuda:0"); H,W=256,128; th=np.deg2rad(np.linspace(-AZD,AZD,W)); NA=84
S=np.exp(1j*np.pi*np.arange(NA)[None,:]*np.sin(th)[:,None]); K=torch.from_numpy(((S.conj()@S.T)/NA).astype(np.complex64)).to(dev)
A=lambda x: torch.abs(torch.matmul(x.to(K.dtype),K.transpose(0,1)))
def tv2(x): return (x[:,1:,:]-x[:,:-1,:]).abs().sum()+(x[:,:,1:]-x[:,:,:-1]).abs().sum()
def solve(y,reg,lam,tv,iters=300,lr=0.05):
    x=(0.01*torch.rand(y.shape,generator=torch.Generator(device='cpu').manual_seed(0))).to(dev).requires_grad_(True); opt=torch.optim.Adam([x],lr=lr)
    for _ in range(iters):
        opt.zero_grad(); r=A(x)-y; loss=(r*r).sum()+lam*(x.abs().sum() if reg=="l1" else (x*x).sum())+tv*tv2(x); loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(min=0)
    return x.detach().cpu().numpy()
EX=[("top1500",1500),("top3000",3000),("top5500",5500),("top8000",8000)]
def run(pp,reg,lam,tv,only=None):
    res={}
    for k in range(0,len(pp),16):
        ch=pp[k:k+16]; Y=torch.from_numpy(np.stack([ra(r) for _,r,_ in ch]).astype(np.float32)).to(dev); X=solve(Y,reg,lam,tv)
        for (fid,r,l),x in zip(ch,X):
            G=gt(l)
            for nm,n in EX:
                if only and nm!=only: continue
                P=topn(x,n,H,W); res.setdefault(nm,{})[fid]=(score(P,G),P)
    return res
val=pairs("val")[::max(1,len(pairs("val"))//40)][:40]; best={}
for reg,lams in [("l1",[0.01,0.05,0.2]),("l2",[0.01,0.1,1.0])]:
    for lam in lams:
        for tv in (0.0,0.1,0.5):
            for nm,d in run(val,reg,lam,tv).items():
                v=np.mean([s["CD"] for s,_ in d.values()]); print("VAL %s lam %g tv %g %s CD %.3f"%(reg,lam,tv,nm,v),flush=True)
                if reg not in best or v<best[reg][0]: best[reg]=(v,lam,tv,nm)
print("CHOSEN",best,flush=True); out={}
test=pairs("test")
for reg,(v,lam,tv,nm) in best.items():
    d=run(test,reg,lam,tv,nm)[nm]; od="points_%s"%reg; os.makedirs(od,exist_ok=True)
    for fid,(s,P) in d.items(): np.concatenate([P,np.zeros((len(P),1))],1).astype(np.float32).tofile(od+"/%s.bin"%fid)
    m={k:float(np.mean([s[k] for s,_ in d.values()])) for k in ["FID","CD","UCD","HD","UMHD","pts"]}; out[reg]=dict(m,lam=lam,tv=tv,extract=nm)
    print("TEST %s: CD %.3f UCD %.3f FID %.1f pts %.0f n=%d"%(reg,m["CD"],m["UCD"],m["FID"],m["pts"],len(d)),flush=True)
json.dump(out,open("kr_l12_test.json","w"),indent=1); print("KR L12 DONE")
