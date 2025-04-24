import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import patches
import time
from scipy.ndimage import zoom
from skimage import filters

class visualizer():
    def __init__(self, tensorWriter=None):
        self.tensorwriter = tensorWriter
        # self.calibMat = np.load('/root/autodl-tmp/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/RADIal/SignalProcessing/CalibrationTable.npy', allow_pickle=True).item()
        # self.PC_CalibMat = np.rollaxis(self.calibMat['Signal'],2,1).reshape(self.calibMat['Signal'].shape[0]*self.calibMat['Signal'].shape[2],self.calibMat['Signal'].shape[1])
        # import pdb;pdb.set_trace()
    
    def visualize_MVradarSRNet(self, 
                               batch_dict,             # 输入+监督
                               ret, # 预测
                               generator = None,
                               descriminator = None,
                               ):
        # dataset: kradar
        lidar_rea_mask, radar_rea_cube = batch_dict['lidar_rea_mask'], batch_dict['radar_rea_cube']
        # 输入        
        ra_np = np.sum(radar_rea_cube.numpy(), axis=2)
        re_np = np.sum(radar_rea_cube.numpy(), axis=3)
        ea_np = np.sum(radar_rea_cube.numpy(), axis=1)
        ra_np = np.squeeze(ra_np[0,...])
        re_np = np.squeeze(re_np[0,...])
        ea_np = np.squeeze(ea_np[0,...])
        
        # 预测
        out_ra = ret['out'][0]
        out_re = ret['out'][1]
        out_ea = ret['out'][2]
        out_ra_np = torch.sigmoid(out_ra).detach().cpu().numpy()
        out_re_np = torch.sigmoid(out_re).detach().cpu().numpy()
        out_ea_np = torch.sigmoid(out_ea).detach().cpu().numpy()
        out_ra_np = np.squeeze(out_ra_np[0,...])
        out_re_np = np.squeeze(out_re_np[0,...])
        out_ea_np = np.squeeze(out_ea_np[0,...])

        # 真值
        gt_ra_np = np.sum(lidar_rea_mask.numpy(), axis=2)
        gt_re_np = np.sum(lidar_rea_mask.numpy(), axis=3)
        gt_ea_np = np.sum(lidar_rea_mask.numpy(), axis=1)
        gt_ra_np = np.squeeze(gt_ra_np[0,...])
        gt_re_np = np.squeeze(gt_re_np[0,...])
        gt_ea_np = np.squeeze(gt_ea_np[0,...])

        self.tensorwriter.add_images("input/range_azimuth", ra_np,dataformats='HW')
        self.tensorwriter.add_images("input/range_elevation", re_np,dataformats='HW')
        self.tensorwriter.add_images("input/elevation_azimuth", ea_np,dataformats='HW')
        self.tensorwriter.add_images("output/range_azimuth", out_ra_np,dataformats='HW')
        self.tensorwriter.add_images("output/range_elevation", out_re_np,dataformats='HW')
        self.tensorwriter.add_images("output/elevation_azimuth", out_ea_np,dataformats='HW')
        self.tensorwriter.add_images("gt/range_azimuth", gt_ra_np,dataformats='HW')
        self.tensorwriter.add_images("gt/range_elevation", gt_re_np,dataformats='HW')
        self.tensorwriter.add_images("gt/elevation_azimuth", gt_ea_np,dataformats='HW')

        # ---------------------
        # 保存模型计算图
        # ---------------------
        # input = batch_dict['radar_rea_cube']
        # x = torch.unsqueeze(input, 1) # b,c,r,e,a
        # # input
        # ra = torch.mean(x, axis=-2).cuda()
        # re = torch.mean(x, axis=-1).cuda()
        # ea = torch.mean(x, axis=-3).cuda()
        # if generator is not None:
        #     self.tensorwriter.add_graph(generator, [ra, re, ea])
        # if descriminator is not None:
        #     self.tensorwriter.add_graph(descriminator, [ra, re, ea])
        # import pdb;pdb.set_trace()

    def visualize_unet3d(self, input, pred, gt):
        # dataset: kradar
        pred = torch.sigmoid(pred)
        input = torch.squeeze(input[0,...])
        gt = torch.squeeze(gt[0,...])
        pred = torch.squeeze(pred[0,...])
        lidar_ra = torch.mean(gt, 1).detach().cpu().numpy()
        # radar_ra = torch.mean(pred, 1).detach().cpu().numpy()
        radar_ra, idx = torch.max(pred, dim=1)
        radar_ra = radar_ra.detach().cpu().numpy()
        pred_numpy = pred.detach().cpu().numpy()
        radar_pc = np.where(pred_numpy>0.5)
        radar_ra_raw = torch.mean(input, 1).detach().cpu().numpy()
        plt.clf()
        plt.imshow(lidar_ra)
        # 存点云图
        gt_numpy = gt.detach().cpu().numpy()
        pc = np.where(gt_numpy==1)
        pc_final = np.concatenate((pc[0].reshape(-1,1), pc[1].reshape(-1,1), pc[2].reshape(-1,1)), axis=1)
        # outname = '/data/yyl-fusion-demo/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/visualization/lidar_ra.pcd'
        # write_pcd_file(outname, pc_final)
        # outname = '/data/yyl-fusion-demo/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/visualization/lidar_ra.png'
        # plt.savefig(outname, dpi=150)
        plt.clf()
        # plt.imshow(radar_ra)
        plt.subplot(1,3,1)
        plt.scatter(pc[2], pc[0], s=1, c='r')
        plt.scatter(radar_pc[2], radar_pc[0], s=1, c='g')

        plt.subplot(1,3,2)
        plt.scatter(pc[1], pc[0], s=1, c='r')
        plt.scatter(radar_pc[1], radar_pc[0], s=1, c='g')

        plt.subplot(1,3,3)
        plt.scatter(pc[2], pc[1], s=1, c='r')
        plt.scatter(radar_pc[2], radar_pc[1], s=1, c='g')
        outname = '/data/yyl-fusion-demo/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/visualization/radar_ra.png'
        plt.savefig(outname, dpi=150)
        plt.clf()
        plt.imshow(radar_ra_raw)
        plt.scatter(pc[2], pc[0], s=1, c='r')
        outname = '/data/yyl-fusion-demo/Object_Detection_with_Super_Resolution_Radar_Point_Cloud_Creation_using_Deep_Learning/visualization/radar_ra_raw.png'
        plt.savefig(outname, dpi=150)
        import pdb;pdb.set_trace()

    def visualize_RADIal(self, batch, ret, fname=None, save=None):
        lidar_ra, radar_ra = batch['lidar_ra'], batch['radar_ra']

        # 输入        
        ra_np = np.squeeze(radar_ra[0,0,...].numpy())
        # 预测
        out_ra = torch.sigmoid(ret['out'])
        out_ra_np = out_ra.detach().cpu().numpy()
        out_ra_np = np.squeeze(out_ra_np[0,...])

        # 真值
        gt_ra_np = np.squeeze(lidar_ra[0,...].numpy())
        on_Tensorboard = save
        if on_Tensorboard:
            self.tensorwriter.add_images("input/range_azimuth", ra_np,dataformats='HW')
            self.tensorwriter.add_images("output/range_azimuth", out_ra_np,dataformats='HW')
            self.tensorwriter.add_images("gt/range_azimuth", gt_ra_np,dataformats='HW')
        else:
            plt.subplot(131)
            plt.imshow(ra_np)
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(out_ra_np)
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(gt_ra_np)
            plt.axis('off')

            viz_path = './viz'
            if not os.path.exists(viz_path):
                os.makedirs(viz_path)
            outname = os.path.join(viz_path, '{:06d}.png'.format(fname))
            plt.savefig(outname, dpi=150)

    def cfar_2d(self, image, guard_width, guard_height, kernel_width, kernel_height, threshold_factor):
        """
        Perform 2D CFAR (Constant False Alarm Rate) detection on an image.
        
        Args:
        image (numpy.ndarray): 2D input image
        guard_width (int): Width of the guard band
        guard_height (int): Height of the guard band
        kernel_width (int): Width of the kernel
        kernel_height (int): Height of the kernel
        threshold_factor (float): Factor to adjust the threshold
        
        Returns:
        numpy.ndarray: Binary image with detected peaks
        """
        rows, cols = image.shape
        result = np.zeros_like(image)

        pad_width = ((kernel_height, kernel_height), (kernel_width, kernel_width))
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

        for i in range(kernel_height, rows + kernel_height):
            for j in range(kernel_width, cols + kernel_width):
                if i - kernel_height < kernel_height or i - kernel_height >= rows - kernel_height or \
                   j - kernel_width < kernel_width or j - kernel_width >= cols - kernel_width:
                    continue  # Skip edge pixels

                kernel = padded_image[i-kernel_height:i+kernel_height+1, j-kernel_width:j+kernel_width+1]
                guard = kernel[kernel_height-guard_height:kernel_height+guard_height+1, 
                               kernel_width-guard_width:kernel_width+guard_width+1]
                
                kernel_sum = np.sum(kernel) - np.sum(guard)
                kernel_area = (2*kernel_height+1)*(2*kernel_width+1) - (2*guard_height+1)*(2*guard_width+1)
                
                noise_level = kernel_sum / kernel_area
                threshold = threshold_factor * noise_level
                
                if padded_image[i, j] > threshold:
                    result[i-kernel_height, j-kernel_width] = 1

        return result

    def save_points_cfar(self, RSP, sample, fname=None, threshold=1.25):
        out_ra = sample
        if out_ra.device.type == 'cuda':
            out_ra_np = out_ra.detach().cpu().numpy()
        else:
            out_ra_np = out_ra.detach().numpy()
        out_ra_np = np.squeeze(out_ra_np.mean(axis=0))
        # out_ra_np = zoom(out_ra_np, zoom=(4, 4), order=3)
        out_ra_np = (out_ra_np-out_ra_np.min())/(out_ra_np.max()-out_ra_np.min())

        # Apply 2D CFAR
        cfar_result = self.cfar_2d(out_ra_np, guard_width=5, guard_height=5, 
                                   kernel_width=15, kernel_height=15, threshold_factor=threshold)

        pixel_indices = np.where(cfar_result > 0)
        real_coords = []
        range_coord = np.linspace(0, out_ra_np.shape[0], out_ra_np.shape[0]) / out_ra_np.shape[0] * 103

        for i in range(len(pixel_indices[0])):
            pixel_y, pixel_x = pixel_indices[0][i], pixel_indices[1][i]
            if pixel_x >= len(RSP.azimuth_coord) or pixel_y >= len(range_coord):
                continue
            azimuth = np.deg2rad(RSP.azimuth_coord[pixel_x])
            range_val = range_coord[pixel_y]
            x = range_val * np.cos(azimuth)
            y = range_val * np.sin(azimuth)
            real_coords.append([x, y, 0])

        real_coords = np.array(real_coords).astype(np.float32)
        if len(real_coords) == 0:
            return np.zeros((1,3))
        return real_coords

    def save_points(self, RSP, sample, fname=None, threshold=0.01):
        # # 预测
        # out_ra = torch.sigmoid(ret['out'])
        out_ra = sample
        if out_ra.device.type == 'cuda':
            out_ra_np = out_ra.detach().cpu().numpy()
        else:
            out_ra_np = out_ra.detach().numpy()
        out_ra_np = np.squeeze(out_ra_np[0,...])
        # Apply cubic interpolation to increase the size of out_ra_np by a factor of 4 in both dimensions
        # out_ra_np = zoom(out_ra_np, zoom=(4, 4), order=3)
        # threshold = 0.01
        threshold = out_ra_np.mean()+0.01
        # pixel_indices = np.where(out_ra_np > threshold)
        # 使用Otsu's方法进行二值化
        if threshold is None:
            threshold = filters.threshold_otsu(out_ra_np)
        binary_image = out_ra_np > threshold
        
        # 使用二值化后的图像来获取像素索引
        pixel_indices = np.where(binary_image)
        real_coords = []
        range_coord = np.linspace(0,out_ra_np.shape[0],out_ra_np.shape[0])/out_ra_np.shape[0]*103
        for i in range(len(pixel_indices[0])):
            pixel_y, pixel_x = pixel_indices[0][i], pixel_indices[1][i]  # y corresponds to range, x to azimuth
            if pixel_x >= len(RSP.azimuth_coord) or pixel_y >= len(range_coord):
                continue
            azimuth = np.deg2rad(RSP.azimuth_coord[pixel_x])  # Convert pixel x to azimuth in radians
            range_val = range_coord[pixel_y]  # Convert pixel y to range
            x = range_val * np.cos(azimuth)  # Convert polar to Cartesian coordinates
            y = range_val * np.sin(azimuth)  # Convert polar to Cartesian coordinates
            real_coords.append([x, y, 0])  # Assuming 0 for the z-coordinate
        real_coords = np.array(real_coords).astype(np.float32)
        # path_root = '/root/autodl-tmp/dataset_with_labels/radars_nanoDiffusion_points'
        # path_root = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/radars_nanoDiffusion_uncond_points'
        # path_root = './eval_radial/points'
        # os.makedirs(path_root, exist_ok=True)
        # pcd_filename = os.path.join(path_root, '{:06d}.bin'.format(fname))
        # print(pcd_filename)
        # if os.path.exists(pcd_filename):
        #     return
        # if len(real_coords) > 0:
        #     real_coords.tofile(pcd_filename)
        #     print(' Frame {:06d} bin file saved!!!!!'.format(fname))
        if len(real_coords) == 0:
            return np.zeros((1,3))
        return real_coords

        # Plotting
        # plt.figure(figsize=(20, 10))
        # plt.style.use('dark_background')  # Set the style to use a dark background
        
        # # Subplot for the point cloud
        # plt.subplot(121)
        # plt.scatter(real_coords[:, 1], real_coords[:, 0], c='white', marker='o')
        # plt.title(f"Point Cloud for Frame {fname:06d}")
        # plt.axis('off')
        # # Subplot for the out_ra_np
        # plt.subplot(122)
        # plt.imshow(out_ra_np, cmap='gray')
        # plt.title(f"Output RA for Frame {fname:06d}")
        # # plt.colorbar()
        
        # # Save the plot
        # plot_filename = os.path.join(path_root, '{:06d}_combined_plot.png'.format(fname))
        # plt.savefig(plot_filename)
        # plt.close()
        # print('figure is saved!!')

        # import pdb;pdb.set_trace()

def write_pcd_file(filename, points):
    with open(filename, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write('COUNT 1 1 1\n')
        f.write('POINTS %d\n'%points.shape[0])
        f.write('DATA ascii\n')
        for p in points:
            # f.write(struct.pack('fff', p[0], p[1], p[2]))
            f.write('%f %f %f\n'%(p[0],p[1],p[2]))


class VisualizerBox():
    def __init__(self):
        pass

    def drawBbox(self, corners, label_width, label_height, label_angle, color='blue'):
        leftup_x, leftup_y = corners[0,0], corners[1,0]
        frontCar_line = corners[:,0:2].T
        plt.gca().add_patch(
            patches.Rectangle((leftup_x, leftup_y), label_width, label_height,
                            angle=label_angle,
                            edgecolor=color,
                            facecolor='none',
                            lw=0.5))
        plt.gca().add_patch(
            patches.Polygon(frontCar_line, closed=False, edgecolor='orange', lw=0.4))


    def rot2D(self, x,y,w,h,theta):
        '''
        :param x:       center x array
        :param y:       center y array
        :param theta:   degree(0-360)
        :param w:       bbox width
        :param h:       bbox height
        :return:        x_arr, y_arr
        '''
        theta = np.pi*theta/180
        rotMat2D = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        inputArr = np.array([x,y]).reshape(2,-1)
        leftUp_corners = np.array([-w/2,-h/2]).reshape(2,-1)
        rightUP_corners = np.array([w/2,-h/2]).reshape(2,-1)
        leftDown_corners = np.array([-w/2,h/2]).reshape(2,-1)
        rightDown_corners = np.array([w/2,h/2]).reshape(2,-1)
        corner_lu = np.dot(rotMat2D,leftUp_corners)+inputArr
        corner_ru = np.dot(rotMat2D,rightUP_corners)+inputArr
        corner_ld = np.dot(rotMat2D,leftDown_corners)+inputArr
        corner_rd = np.dot(rotMat2D,rightDown_corners)+inputArr
        corner_lu = corner_lu.reshape(-1)
        corner_ru = corner_ru.reshape(-1)
        corner_ld = corner_ld.reshape(-1)
        corner_rd = corner_rd.reshape(-1)
        corners = np.array([corner_lu[0], corner_ru[0], corner_ld[0], corner_rd[0],corner_lu[1], corner_ru[1], corner_ld[1], corner_rd[1]]).reshape(2,-1)
        return corners

    def gtDraw(self, lidarpc, radar, bboxes, color='green'):
        # plt.figure(facecolor='black')
        if radar != []:
            plt.imshow(radar,cmap='gray')
        if lidarpc != []:
            x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
            # show lidar point cloud
            plt.scatter(x_arr, y_arr, s=0.05, c='white')
        if len(bboxes)==0:
            return
        # show bbox
        ids, center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4], bboxes[:,5]
        num = len(center_x)
        for i in range(num):
            id = ids[i]
            center_x_ = center_x[i]
            center_y_ = center_y[i]
            label_w_ = label_width[i]
            label_h_ = label_height[i]
            label_a_ = label_angle[i]
            corners = self.rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
            self.drawBbox(corners, label_w_, label_h_, label_a_, color)

    def predDraw(self, lidarpc, radar, bboxes, color='red'):
        if radar != []:
            plt.imshow(radar,cmap='gray')
        if lidarpc != []:
            x_arr, y_arr = lidarpc[:,0], lidarpc[:,1]
            # show lidar point cloud
            plt.scatter(x_arr, y_arr, s=0.05, c='white')
        if len(bboxes)==0:
            return
        # show bbox
        center_x, center_y, label_width, label_height, label_angle = bboxes[:,0], bboxes[:,1], bboxes[:,3], bboxes[:,4], bboxes[:,5]
        num = len(center_x)
        for i in range(num):
            center_x_ = center_x[i]
            center_y_ = center_y[i]
            label_w_ = label_width[i]
            label_h_ = label_height[i]
            label_a_ = label_angle[i]
            corners = self.rot2D(center_x_, center_y_, label_w_, label_h_, label_a_)
            self.drawBbox(corners, label_w_, label_h_, label_a_, color)

    def drawTxt(self, predBox, pr_table):
        num = predBox.shape[0]
        for i in range(num):
            x = predBox[i,0]
            y = predBox[i,1]
            iou = pr_table[i,1]
            plt.text(x, y, '{:.2f}'.format(iou), bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)

