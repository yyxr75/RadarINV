"""Fig. 6 (v2): cross-scenario, both priors retrained with one recipe; two frames per scenario, chosen as the two
closest to the scenario median CD of the All prior. Points = the scored points (top8000 + radius outlier r=0.5,k=2)."""
import numpy as np, json, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
from ours_pp_lib import pts, ror, load
SP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
D = os.environ["RADIAL_ROOT"].rstrip("/") + "/"
S = json.load(open(SP + "tab5/subset.json")); A = json.load(open(SP + "tab5/allho.json"))
C = json.load(open(SP + "tab5/prior_csfinal.json")); C.update(json.load(open(SP + "tab5/cs_city_ho.json")))
rows = [("City", "city_heldout", "cs_city_ho"), ("Countryside", "countryside", "prior_csfinal"), ("Highway", "highway", "prior_csfinal")]
pick = {}
for name, key, _ in rows:
    f = ["%06d" % i for i in S[key]]; cd = np.array([A[x]["CD"] for x in f]); med = np.median(cd)
    pick[name] = sorted(f, key=lambda x: abs(A[x]["CD"] - med))[:2]
print({k: [(x, round(A[x]["CD"], 2), round(C[x]["CD"], 2)) for x in v] for k, v in pick.items()})
AZ = np.load(SP + "../CalibrationTable.npy", allow_pickle=True).item()["Azimuth_table"]
RE, TE = np.meshgrid(np.arange(513) / 512 * 103.0, np.deg2rad(np.interp(np.arange(769) / 768 * (len(AZ) - 1), np.arange(len(AZ)), AZ)), indexing="ij")   # as Fig. 4
def radb(f): m = np.maximum(np.fromfile(D + "radars_ra_interp/%s.bin" % f).reshape(512, 768), 1e-12); return 10 * np.log10(m / np.median(m))
VMAX = np.percentile(np.concatenate([radb(f).ravel() for v in pick.values() for f in v]), 99.9)
def P(run, f): return ror(pts(load(SP + "tab5/%s/%s/%s_decoded.npy" % (run, f, f)), "top8000"), 0.5, 2)
cols = ["Camera", "Radar RA Input", "All\nas Prior", "Countryside\nas Prior"]
fig, ax = plt.subplots(3, 8, figsize=(14, 5.4))
for r, (name, key, csrun) in enumerate(rows):
    for g, f in enumerate(pick[name]):
        G = np.fromfile(D + "lidars_no_ground/%s.bin" % f, np.float32).reshape(-1, 3)[:, :2]
        for c in range(4):
            a = ax[r, 4 * g + c]; a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.5); s.set_edgecolor("#b3b3b3")
            if c == 0: a.imshow(Image.open(D + "images/%s.png" % f)); a.set_aspect("auto")
            elif c == 1:   # drawn on the BEV plane like the point panels, dB above the frame median, one shared window
                a.pcolormesh(RE * np.sin(TE), RE * np.cos(TE), radb(f), cmap="viridis", vmin=0, vmax=VMAX, shading="flat", rasterized=True)
                a.set_xlim(-60, 60); a.set_ylim(0, 103); a.set_aspect("auto")
            else:
                Q = P("allho" if c == 2 else csrun, f); cd = (A if c == 2 else C)[f]["CD"]
                a.scatter(G[::3, 1], G[::3, 0], s=0.2, c="blue", lw=0, rasterized=True)
                a.scatter(Q[:, 1], Q[:, 0], s=0.6, c="red", lw=0, rasterized=True)
                a.set_xlim(-60, 60); a.set_ylim(0, 103); a.set_aspect("auto")
                a.text(0.97, 0.03, "CD %.2f" % cd, transform=a.transAxes, ha="right", va="bottom", fontsize=7)
        if g == 0: ax[r, 0].set_ylabel(name, fontsize=11, fontweight="bold")
plt.subplots_adjust(left=0.025, right=0.995, top=0.90, bottom=0.005, wspace=0.04, hspace=0.05)
for c in range(8):
    b = ax[0, c].get_position(); fig.text((b.x0 + b.x1) / 2, 0.95, cols[c % 4], ha="center", va="center", fontsize=9, fontweight="bold")
plt.savefig(SP + "fig6_bev.pdf", dpi=220); plt.savefig(SP + "fig6_bev.png", dpi=110); print("saved")
