import numpy as np, os
from PIL import Image
SP=os.path.dirname(os.path.abspath(__file__)); K=os.environ["KRADAR_ROOT"].rstrip("/") + "/" + "1/"
out=SP+"/radardiff_kradar_fig/test/"; rd,ld=out+"range_azimuth_heatmap/",out+"lidar_pcl_bev_polar_img/"
for d in (rd,ld): os.makedirs(d,exist_ok=True)
for cube,lid in [(203,172),(248,217),(258,227),(272,241),(365,334),(322,291)]:
    fid="1_%05d"%cube
    ra=np.fromfile(K+"radar_rea_cube/cube_%05d.bin"%cube,np.float32).reshape(256,32,128).mean(1); ra=(ra-ra.min())/(ra.max()-ra.min()+1e-12)
    Image.fromarray((ra*255).astype(np.uint8),"L").save(rd+fid+".png")
    mm=np.fromfile(K+"lidar_rea_mask/os2-64_%05d.bin"%lid,np.float32).reshape(256,32,128).mean(1); m=mm>mm.mean()+0.01
    Image.fromarray(m.astype(np.uint8)*255,"L").save(ld+fid+".png")
print("prepared", len(os.listdir(rd)))
