"""Consolidated real-radar test table (424 frames, eval_points protocol) + paired bootstrap of Ours vs the fair L1/L2."""
import numpy as np, glob, os, sys, json
sys.path.insert(0,".."); from eval_points import score
D=os.environ["RADIAL_ROOT"].rstrip("/") + "/"; AZ=np.load("../../CalibrationTable.npy",allow_pickle=True).item()["Azimuth_table"]
test=sorted(set(open(D+"ImageSets/test.txt").read().split()))
def ld(p,c): a=np.fromfile(p,np.float32); return a.reshape(-1,c)[:,:2] if a.size else np.zeros((0,2))
rows=[("CFAR (stored)",D+"radars_pcd",4),("L1, real radar (val-tuned)","points_l1",3),("L2, real radar (val-tuned)","points_l2",3),
      ("RadarHD (stored, supervised)",D+"radars_unet",3),("Diffradar (re-run, supervised)","../diffradar/test_diffradar/radar_points",3),
      ("Radar-Diffusion (retrained, supervised)","../radardiff_runs/points_final_test",3),("Ours, released code, real radar, top-5500","points_ours_released",3),("Ours final","points_ours_pp",3),("Ours (stored outputs)",D+"radars_radarINV",3)]
per={}; out={}
for name,d,c in rows:
    s=[]
    for f in test:
        p=os.path.join(d,f+".bin")
        if not os.path.exists(p): continue
        P=ld(p,c)
        if len(P)<2: continue
        r=score(P,ld(D+"lidars_no_ground/%s.bin"%f,3)); s.append(r); per.setdefault(name,{})[f]=r["CD"]
    m={k:float(np.mean([x[k] for x in s])) for k in ["FID","CD","UCD","MHD","pts"]}; out[name]=dict(m,n=len(s))
    print("%-44s FID %6.1f  CD %.3f  UCD %.3f  MHD %.3f  pts %7.0f  n=%d"%(name,m["FID"],m["CD"],m["UCD"],m["MHD"],m["pts"],len(s)))
ours=per["Ours final"]
for b in ["L1, real radar (val-tuned)","L2, real radar (val-tuned)","RadarHD (stored, supervised)","Radar-Diffusion (retrained, supervised)"]:
    fs=[f for f in ours if f in per[b]]; d=np.array([ours[f]-per[b][f] for f in fs]); rng=np.random.default_rng(0); bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)]
    print("Ours(released) - %-38s %+.3f  95%% CI [%+.3f, %+.3f]  Ours better on %.0f%% of %d frames"%(b,d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),100*(d<0).mean(),len(d)))
json.dump(out,open("final_table_test.json","w"),indent=1)
