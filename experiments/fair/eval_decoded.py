"""Score sampler outputs with the Table III protocol for Ours (top8000 + radius-outlier removal r=0.5, k=2; GT = lidars_no_ground).
usage: eval_decoded.py <out_json> <glob of *_decoded.npy> [more globs]"""
import numpy as np, glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
from ours_pp_lib import pts, ror, load
sys.path.insert(0, ".."); from eval_points import score
D = os.environ["RADIAL_ROOT"].rstrip("/") + "/"; res = {}
for g in sys.argv[2:]:
    for p in sorted(glob.glob(g)):
        f = os.path.basename(p)[:6]; P = ror(pts(load(p), "top8000"), 0.5, 2)
        s = score(P, np.fromfile(D + "lidars_no_ground/%s.bin" % f, np.float32).reshape(-1, 3)[:, :2])
        res[f] = {k: float(s[k]) for k in ["FID", "CD", "UCD", "MHD", "UMHD", "pts"]}
json.dump(res, open(sys.argv[1], "w"), indent=0)
print(len(res), "frames; mean CD %.3f" % np.mean([r["CD"] for r in res.values()]) if res else "no frames")
