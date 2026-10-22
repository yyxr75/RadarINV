import numpy as np, torch, torch.nn.functional as F, os
from kr_common import *
os.makedirs("ours_in/radars_ra_interp",exist_ok=True); sub=pairs("test")[::8]
for fid,r,l in sub:
    a=torch.from_numpy(ra(r).astype(np.float32))[None,None]; b=F.interpolate(a,(512,768),mode="bilinear",align_corners=False)[0,0].numpy().astype(np.float64)
    b.tofile("ours_in/radars_ra_interp/%s.bin"%fid)
open("subset_ids.txt","w").write("\n".join(f for f,_,_ in sub)); print(len(sub),"K-Radar test frames prepared")
