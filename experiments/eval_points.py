"""Score a directory of point clouds with the paper's stored evaluation protocol.

GT is the uncropped ground-removed LiDAR. Distances are first-power Euclidean
nearest-neighbour distances on the (x, y) coordinates, in meters, with
a = predicted->GT and b = GT->predicted. The metric set follows Luan et al.,
ICRA 2024: CD = mean(a) + mean(b), MHD = max(median(a), median(b)) (as in
RadarHD), and the unidirectional UCD = mean(b) and UMHD = median(b), measured
from the LiDAR to the predicted point cloud only. FID_BEV is the Frechet
distance between Gaussian fits to the (x, y) coordinates of the two point sets.  Usage: eval_points.py <points_dir> <frame_list.txt> [label]
"""
import sys, os, json, numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import sqrtm
D = os.environ["RADIAL_ROOT"].rstrip("/") + "/"

def frechet(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0); Sa, Sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm = sqrtm(Sa.dot(Sb)); cm = cm.real if np.iscomplexobj(cm) else cm
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(Sa + Sb - 2 * cm))

def score(P, G):
    a = cKDTree(G).query(P)[0]; b = cKDTree(P).query(G)[0]
    return dict(CD=a.mean() + b.mean(), UCD=b.mean(), MHD=max(np.median(a), np.median(b)),
                UMHD=np.median(b), FID=frechet(P, G), pts=len(P))

def load_pts(path, cols):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(-1, cols)[:, :2] if a.size else np.zeros((0, 2), np.float32)

def evaluate(points_dir, frames, cols=3):
    per = {}
    for f in frames:
        p = os.path.join(points_dir, f + ".bin")
        if not os.path.exists(p): continue
        P = load_pts(p, cols)
        if len(P) < 2: continue
        G = np.fromfile(D + "lidars_no_ground/%s.bin" % f, dtype=np.float32).reshape(-1, 3)[:, :2]
        per[f] = score(P, G)
    return per

def summarise(per):
    keys = ["FID", "CD", "UCD", "MHD", "UMHD", "pts"]
    out = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}; return dict(out, n=len(per))

if __name__ == "__main__":
    pdir, flist = sys.argv[1], sys.argv[2]; label = sys.argv[3] if len(sys.argv) > 3 else pdir
    frames = sorted(f[:-4] for f in os.listdir(pdir) if f.endswith(".bin")) if flist == "-" else sorted(set(open(flist).read().split()))
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(pdir.rstrip("/")), "eval.json")
    per = evaluate(pdir, frames, cols=4 if "radars_pcd" in pdir else 3)
    s = summarise(per)
    print("%-28s n=%-4d FID %8.2f  CD %6.3f  UCD %6.3f  MHD %6.3f  UMHD %6.3f  pts %7.0f" % (
        label, s["n"], s["FID"], s["CD"], s["UCD"], s["MHD"], s["UMHD"], s["pts"]))
    json.dump({"summary": s, "per_frame": per}, open(out, "w"), indent=1)
