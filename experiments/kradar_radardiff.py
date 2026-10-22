"""K-Radar test split (367 frames) for a zero-shot run of Radar-Diffusion, plus conversion and scoring.

prep <variant>   : write <SP>/radardiff_kradar_<variant>/test/{range_azimuth_heatmap,lidar_pcl_bev_polar_img}/<id>.png
                   radar = elevation-mean of the 256x32x128 cube (values are in dB), per-frame min-max to [0,1]
                   on the dB scale ('db'), on magnitude 10^(dB/20) ('mag') or on power 10^(dB/10) ('pow');
                   lidar target and GT = elevation-mean of the LiDAR polar mask thresholded at mean+0.01 (the paper's save_points_kradar rule). GT points -> <SP>/kradar_gt/<id>.bin.
topoints <png_dir> <out_dir> : polar output PNG (pixel > 64) -> points in the K-Radar frame
                   (rows: range 0..118.037 m, cols: azimuth -53..53 deg, as in the paper's save_points_kradar).
eval <points_dir> <label> <out.json> : the paper's protocol (eval_points.score) against kradar_gt.
"""
import sys, os, json, glob, numpy as np
from PIL import Image
SP = os.path.dirname(os.path.abspath(__file__)); K = os.environ["KRADAR_ROOT"].rstrip("/") + "/"
RMAX, AZ = 118.037, 53.0
sys.path.insert(0, SP); from eval_points import score, summarise

def polar_to_pts(mask, H, W):
    iy, ix = np.where(mask); r = (iy + 0.5) / H * RMAX; az = np.deg2rad(-AZ + (ix + 0.5) / W * 2 * AZ)
    return np.stack([r * np.cos(az), r * np.sin(az), np.zeros_like(r)], 1).astype(np.float32)

def pairs():
    for line in open(K + "testset.txt"):
        r, l = line.split(); r = K + r.rsplit("/data/", 1)[1]; l = K + l.rsplit("/data/", 1)[1]
        yield r.split("/")[-3] + "_" + os.path.basename(r)[5:-4], r, l   # id = <seq>_<cube number>

if sys.argv[1] == "prep":
    v = sys.argv[2]; out = SP + "/radardiff_kradar_%s/test/" % v
    rd, ld, gd = out + "range_azimuth_heatmap/", out + "lidar_pcl_bev_polar_img/", SP + "/kradar_gt/"
    for d in (rd, ld, gd): os.makedirs(d, exist_ok=True)
    for fid, r, l in pairs():
        ra = np.fromfile(r, np.float32).reshape(256, 32, 128).mean(1)
        ra = {"db": ra, "mag": 10 ** (ra / 20), "pow": 10 ** (ra / 10)}[v]
        ra = (ra - ra.min()) / (ra.max() - ra.min() + 1e-12)
        Image.fromarray((ra * 255).astype(np.uint8), "L").save(rd + fid + ".png")
        mm = np.fromfile(l, np.float32).reshape(256, 32, 128).mean(1); m = mm > mm.mean() + 0.01   # the paper's save_points_kradar rule
        Image.fromarray(m.astype(np.uint8) * 255, "L").save(ld + fid + ".png")
        polar_to_pts(m, 256, 128).tofile(gd + fid + ".bin")
    print(v, "prepared", len(os.listdir(rd)), "frames")
elif sys.argv[1] == "topoints":
    src, dst = sys.argv[2], sys.argv[3]; os.makedirs(dst, exist_ok=True); n = []
    for f in sorted(glob.glob(src + "/*.png")):
        a = np.array(Image.open(f).convert("L")); P = polar_to_pts(a > 64, *a.shape)
        P.tofile(os.path.join(dst, os.path.basename(f)[:-4] + ".bin")); n.append(len(P))
    print("%d frames -> %s  mean %.0f pts (min %d max %d)" % (len(n), dst, np.mean(n), min(n), max(n)))
elif sys.argv[1] == "eval":
    pdir, label, out = sys.argv[2:5]; per = {}
    for f in sorted(os.listdir(pdir)):
        P = np.fromfile(os.path.join(pdir, f), np.float32).reshape(-1, 3)[:, :2]
        G = np.fromfile(SP + "/kradar_gt/" + f, np.float32).reshape(-1, 3)[:, :2]
        if len(P) >= 2 and len(G) >= 2: per[f[:-4]] = score(P, G)
    s = summarise(per)
    print("%-28s n=%-4d FID %8.2f  CD %6.3f  UCD %6.3f  HD %8.2f  UMHD %7.2f  pts %7.0f" % (label, s["n"], s["FID"], s["CD"], s["UCD"], s["HD"], s["UMHD"], s["pts"]))
    json.dump({"summary": s, "per_frame": per}, open(out, "w"), indent=1)
