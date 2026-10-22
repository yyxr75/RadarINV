"""RADIal -> Radar-Diffusion on-disk format.

One folder per split, each with range_azimuth_heatmap/<id>.png (radar RA heatmap,
8-bit) and lidar_pcl_bev_polar_img/<id>.png (LiDAR polar BEV occupancy, 0/255).
Both are written at RADIal's native 512x768 polar grid; the baseline's own
transform resizes them to 128x128, exactly as it does for ColoRadar.
"""
import os, numpy as np
from PIL import Image
D = os.environ["RADIAL_ROOT"].rstrip("/") + "/"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/radardiff_data/"
H, W = 512, 768
for split in ["train", "val", "test"]:
    ids = sorted(set(open(D + "ImageSets/%s.txt" % split).read().split()))
    rd = OUT + split + "/range_azimuth_heatmap/"; ld = OUT + split + "/lidar_pcl_bev_polar_img/"
    os.makedirs(rd, exist_ok=True); os.makedirs(ld, exist_ok=True)
    n = 0
    for f in ids:
        ra = np.fromfile(D + "radars_ra_interp/%s.bin" % f).reshape(H, W)
        Image.fromarray(np.clip(ra * 255, 0, 255).astype(np.uint8), "L").save(rd + f + ".png")
        m = np.fromfile(D + "lidars_mask/%s.bin" % f).reshape(H, W)
        Image.fromarray((m > 0).astype(np.uint8) * 255, "L").save(ld + f + ".png")
        n += 1
    print(split, n, "frames")
