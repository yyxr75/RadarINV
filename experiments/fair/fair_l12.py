"""Real-radar L1/L2 baselines, fair protocol: calibrated 192-ch operator (dataset window) at 512x768, projected Adam 300 it, x>=0.
Tune (lambda, extraction in {otsu, top-1500/3000/5500/8000}) on 80 val frames by CD; run the chosen config on all 424 test frames.
Save points to fair/points_<l1|l2>/ and per-frame CD."""
import numpy as np, torch, sys, os, json
sys.argv=[sys.argv[0],"x","x","12","16","x"]; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open("reg_calib.py").read().split("frames=open(sys.argv[1])")[0])
OUT=os.path.abspath("fair"); D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; K=K_n(12,16)
def ex(m):
    v=np.sort(m.ravel()); return [("otsu",None)]+[("top%d"%n,v[-n]) for n in (1500,3000,5500,8000)]
def run(frames,reg,lam,extract=None):
    per={}
    for k in range(0,len(frames),8):
        ch=frames[k:k+8]; Y=torch.stack([torch.from_numpy(np.fromfile(D+"radars_ra_interp/%s.bin"%f).reshape(H,W)).float() for f in ch]).to(dev); X=solve(Y,K,reg,lam)
        for f,x in zip(ch,X):
            G=np.fromfile(D+"lidars_no_ground/%s.bin"%f,np.float32).reshape(-1,3)[:,:2]
            for name,th in ex(x):
                if extract and name!=extract: continue
                P=to_xy(x,th); per.setdefault(name,{})[f]=(score(P,G) if len(P)>=2 else None,P)
    return per
val=sorted(set(open(D+"ImageSets/val.txt").read().split())); val=val[::max(1,len(val)//80)][:80]
best={}
for reg,lams in [("l1",[0.002,0.01,0.05,0.2]),("l2",[0.01,0.05,0.2,1.0])]:
    for lam in lams:
        for name,pp in run(val,reg,lam).items():
            v=np.mean([s["CD"] for s,_ in pp.values() if s]); print("VAL %s lam %g %-8s CD %.3f"%(reg,lam,name,v),flush=True)
            if reg not in best or v<best[reg][0]: best[reg]=(v,lam,name)
print("CHOSEN",best,flush=True)
test=sorted(set(open(D+"ImageSets/test.txt").read().split())); res={}
for reg,(v,lam,name) in best.items():
    pp=run(test,reg,lam,name)[name]; od=OUT+"/points_%s"%reg; os.makedirs(od,exist_ok=True)
    for f,(s,P) in pp.items(): np.concatenate([P,np.zeros((len(P),1))],1).astype(np.float32).tofile(od+"/%s.bin"%f)
    m={k:float(np.mean([s[k] for s,_ in pp.values() if s])) for k in ["FID","CD","UCD","MHD","pts"]}; res[reg]=dict(m,lam=lam,extract=name)
    print("TEST %s (lam %g, %s): CD %.3f UCD %.3f FID %.1f pts %.0f"%(reg,lam,name,m["CD"],m["UCD"],m["FID"],m["pts"]),flush=True)
json.dump(res,open(OUT+"/fair_l12_test.json","w"),indent=1); print("FAIR L12 DONE")
