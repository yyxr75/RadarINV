"""Radar-Diffusion polar output (128x128 PNG) -> RADIal-frame point clouds (.bin, Nx3 float32).

Threshold follows the baseline's own extraction (polar_image_to_pcl: pixel > 64 of 255).
Geometry follows RADIal, because the model was trained on RADIal's polar grid: rows are range
0..103 m from the top, columns are azimuth on the calibrated table (-75..75 deg).
"""
import sys, os, glob, numpy as np
from PIL import Image
SP = os.path.dirname(os.path.abspath(__file__))
AZ = np.load(SP + "/../CalibrationTable.npy", allow_pickle=True).item()["Azimuth_table"]
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); n = 0; pts = []
for f in sorted(glob.glob(src + "/*.png")):
    a = np.array(Image.open(f).convert("L")); H, W = a.shape
    iy, ix = np.where(a > 64)
    r = (iy + 0.5) / H * 103.0
    az = np.deg2rad(np.interp((ix + 0.5) / W * (len(AZ) - 1), np.arange(len(AZ)), AZ))
    P = np.stack([r * np.cos(az), r * np.sin(az), np.zeros_like(r)], 1).astype(np.float32)
    fid = os.path.basename(f)[:-4]
    P.tofile(os.path.join(dst, fid + ".bin")); n += 1; pts.append(len(P))
print("%d frames -> %s   mean %.0f pts/frame (min %d, max %d)" % (n, dst, np.mean(pts), min(pts), max(pts)) if n else "no frames found in " + src)
