from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import numpy as np
import torch
import pickle
from scipy.ndimage import zoom

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='celeb')
class CELEBDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    

@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    

# dataset_rootPath = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/'
# dataset_rootPath = '/data/yyl-fusion-demo/dataset_with_labels/'
# dataset_rootPath = '/root/autodl-tmp/dataset_with_labels/'
# dataset_rootPath = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/'
dataset_rootPath = '/scratch/project/cpautodriving/yanlongyang/dataset_with_labels/'
@register_dataset(name='DatasetRADIal')
class DatasetRADIal(torch.utils.data.Dataset):
    def __init__(
        self, 
        root, 
        RBINS=512,
        ABINS=768, 
        M=0,
    ):
        with open(root, 'rb') as f:
            dataset_infos = pickle.load(f)

        self.infos = dataset_infos # radar_file_name / lidar_file_name
        self.RBINS = RBINS
        self.ABINS = ABINS
        self.history = M

        # lidar_files = sorted(glob.glob(self.lidar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        # radar_files = sorted(glob.glob(self.radar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        lidar_files = []
        radar_files = []
        self.targets_infos = []
        for info in self.infos:
            radar_file = dataset_rootPath+info['radar']['radars_ra_path']
            lidar_file = dataset_rootPath+info['point_cloud']['lidars_path']
            lidar_files.append(lidar_file)
            radar_files.append(radar_file)

            locs = info['annos']['location'].reshape(-1,3)
            dims = info['annos']['dimensions'].reshape(-1,3)
            rots = info['annos']['rotation_y'].reshape(-1,1)

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            self.targets_infos.append(gt_boxes)


        self.labels = lidar_files
        self.input_data = []

        # ----------------
        # 以radar连续多帧作为输入
        # ----------------
        trajs = []
        if self.history == 0:
            self.labels = lidar_files
            self.input_data = radar_files
        else:
            for idx, radar_file in enumerate(radar_files):
                sequence = [radar_file]  # Start with the current file
                file_index = int(radar_file.split('/')[-1].split('.')[0])  # Extract the file index
                expected_index = file_index - 1  # The index we expect to find next
                for _ in range(self.history - 1):  # We already have one file, so start one less
                    found = False
                    for search_idx in range(idx - 1, -1, -1):  # Search backwards for previous files
                        search_file = radar_files[search_idx]
                        search_index = int(search_file.split('/')[-1].split('.')[0])
                        if search_index == expected_index:
                            sequence.insert(0, search_file)  # Insert at the beginning
                            expected_index -= 1  # Update the expected index for the next iteration
                            found = True
                            break
                    if not found:  # If we didn't find a consecutive file
                        sequence.insert(0, sequence[0])  # Duplicate the first file in the sequence
                self.input_data.append(sequence)


    def __len__(self):
        return len(self.input_data)

    def __filenames__(self):
        return [x[-36:-17]+x[-10:-4] for x in self.labels]

    def get_lidar(self, label_filename):

        # a = Image.open(label_filename)
        parts = label_filename.split('/')
        parts[-2] = 'lidars_mask'
        label_filename = '/'.join(parts)
        a = np.fromfile(label_filename).reshape(512,768)
        if self.RBINS != 512 or self.ABINS != 768:
            zoom_factors = (self.RBINS / a.shape[0], self.ABINS / a.shape[1])
            a = zoom(a, zoom_factors, order=1)  # order=1 for bilinear interpolation
        y = torch.Tensor(np.reshape(a, (1,self.RBINS,self.ABINS)))
        return y

    def get_radar(self, input_filename):
        # a = Image.open(input_filename)
        parts = input_filename.split('/')
        parts[-2] = 'radars_ra_fullscale_norm'
        input_filename = '/'.join(parts)
        a = np.fromfile(input_filename).reshape(512,768)
        if self.RBINS != 512 or self.ABINS != 768:
            zoom_factors = (self.RBINS / a.shape[0], self.ABINS / a.shape[1])
            a = zoom(a, zoom_factors, order=1)  # order=1 for bilinear interpolation
        # ra = (a-a.min())/(a.max()-a.min()) # 去掉归一化，因为用了所有数据的最大最小值来做
        X = torch.Tensor(np.reshape(a, (1,self.RBINS,self.ABINS)))
        return X

    def get_radar_pcd(self, input_filename):
        parts = input_filename.split('/')
        parts[-2] = 'radars_pcd_mask'
        input_filename = '/'.join(parts)
        radars_pcd_data = np.fromfile(input_filename).reshape(1, 512, 768)  # Assuming 4 columns for radars_pcd data
        radars_pcd_data = torch.Tensor(radars_pcd_data)
        return radars_pcd_data

    def __getitem__(self, index):
        # Select sample
        if self.history == 0:
            input_filename = self.input_data[index]
            label_filename = self.labels[index]
            X, y = self.get_radar(input_filename), self.get_lidar(label_filename)
            # X, y = self.get_radar_pcd(input_filename), self.get_lidar(label_filename)
        else:
            X = torch.Tensor([])
            input_filenames = self.input_data[index]
            label_filename = self.labels[index]
            for i in input_filenames:
                xx = self.get_radar(i)
                # xx = self.get_radar_pcd(i)
                X = torch.cat((X, xx), dim=0)        
            y = self.get_lidar(label_filename)
        batch_dict  = {}
        # X = torch.nn.functional.interpolate(X.unsqueeze(0), size=(128,192), mode='area').squeeze(0)
        # y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(128,192), mode='area').squeeze(0)
        batch_dict['radar_ra'] = torch.cat([X, X, X], dim=0)
        batch_dict['lidar_ra'] = torch.cat([y, y, y], dim=0)
        # box: x,y,z,w,h,l,angle
        # batch_dict['target'] = torch.tensor(self.targets_infos[index]) # 增加第一维为了在batch collate能够合并不同数量的目标（其实目标框暂时没用）
        batch_dict['data_fname'] = torch.tensor(int(label_filename.split('/')[-1].split('.')[0]))
        return batch_dict
    
def radial_collate_fn(batch):
    """
    Custom collate function to handle batches where the number of targets can vary between samples.
    """
    batch_dict = {}
    # Initialize empty lists to hold the batched data
    batch_dict['radar_ra'] = []
    batch_dict['lidar_ra'] = []
    batch_dict['target'] = []
    batch_dict['data_fname'] = []

    # Iterate through each sample in the batch
    for item in batch:
        batch_dict['radar_ra'].append(item['radar_ra'])
        batch_dict['lidar_ra'].append(item['lidar_ra'])
        batch_dict['target'].append(torch.tensor(item['target'], dtype=torch.float))
        batch_dict['data_fname'].append(item['data_fname'])

    # Stack the radar and lidar data since they have consistent dimensions across samples
    batch_dict['radar_ra'] = torch.stack(batch_dict['radar_ra'], dim=0)
    batch_dict['lidar_ra'] = torch.stack(batch_dict['lidar_ra'], dim=0)

    # The target data might have varying dimensions, so we leave it as a list
    # Convert data_fname to a tensor for consistent handling
    batch_dict['data_fname'] = torch.tensor(batch_dict['data_fname'], dtype=torch.int64)

    return batch_dict


# Creates a dataloader in batches for training and testing

import os
from PIL import Image
import numpy as np
import torch
import glob
@register_dataset(name='DatasetRadarHD')
class DatasetRadarHD(torch.utils.data.Dataset):

    def __init__(self, root, sub,
                RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
                RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=1024, M=0):

        self.basepath = root
        self.lidar_path = self.basepath + sub + '/lidar/*'
        self.radar_path = self.basepath + sub + '/radar/*'

        self.RBINS = RBINS
        self.ABINS_RADAR = ABINS_RADAR
        self.ABINS_LIDAR = ABINS_LIDAR
        self.RBINS_ORIG = RBINS_ORIG
        self.ABINS_RADAR_ORIG = ABINS_RADAR_ORIG
        self.ABINS_LIDAR_ORIG = ABINS_LIDAR_ORIG
        self.history = M

        lidar_files = sorted(glob.glob(self.lidar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        radar_files = sorted(glob.glob(self.radar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        
        if self.history == 0:
            self.labels = lidar_files
            self.input_data = radar_files
        else:
            traj = [int(os.path.basename(x).split('_')[1]) for x in lidar_files]
            time_st = [int(os.path.basename(x).split('_')[2].split('.')[0]) for x in lidar_files]
            self.labels = []
            self.input_data = []

            for i in np.unique(traj):
                start_idx = np.where(traj==i)[0][0]
                end_idx = np.where(traj==i)[0][-1]+1
                print("Traj ", i, "Time ", time_st[start_idx], " ", time_st[end_idx-1])
                radar_files_time = radar_files[start_idx:end_idx]
                lidar_files_time = lidar_files[start_idx:end_idx]

                x_local = []
                for j in range(self.history, len(radar_files_time)):
                    x_local.append(radar_files_time[j-self.history:j+1])
                y_local = lidar_files_time[self.history:]
                
                self.labels.extend(y_local)
                self.input_data.extend(x_local)

    def __len__(self):
        return len(self.input_data)

    def __filenames__(self):
        return [os.path.basename(x).split('.')[0].split('L_')[1] for x in self.labels]

    def get_lidar(self, label_filename):

        a = Image.open(label_filename)
        y = torch.Tensor(np.reshape(np.asarray(a,dtype=np.bool_), (1,self.RBINS_ORIG,self.ABINS_LIDAR_ORIG)))
        y = y[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_LIDAR_ORIG/self.ABINS_LIDAR)]

        # Resize y from (256, 512) to (256, 64)
        y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(self.RBINS, self.ABINS_RADAR), mode='nearest').squeeze(0)

        return y

    def get_radar(self, input_filename):

        a = Image.open(input_filename)
        X = torch.Tensor(np.reshape(np.asarray(a)/255.0, (1,self.RBINS_ORIG,self.ABINS_RADAR_ORIG)))
        X = X[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_RADAR_ORIG/self.ABINS_RADAR)]

        return X

    def __getitem__(self, index):

        # Select sample
        if self.history == 0:
            input_filename = self.input_data[index]
            label_filename = self.labels[index]
            radar_data, lidar_data = self.get_radar(input_filename), self.get_lidar(label_filename)
        
        else:
            radar_data = torch.Tensor([])
            input_filenames = self.input_data[index]
            label_filename = self.labels[index]
            for i in input_filenames:
                xx = self.get_radar(i)
                radar_data = torch.cat((radar_data, xx), dim=0)        
            lidar_data = self.get_lidar(label_filename)
        batch_dict = {}
        # print(label_filename.split('_')[-2]+'_'+label_filename.split('_')[-1].split('.')[0])
        
        radar_data = torch.cat([radar_data, radar_data, radar_data], dim=0)
        lidar_data = torch.cat([lidar_data, lidar_data, lidar_data], dim=0)
        
        batch_dict['radar_ra'] = radar_data
        batch_dict['lidar_ra'] = lidar_data
        batch_dict['data_fname'] = label_filename.split('_')[-2] + '_' + label_filename.split('_')[-1].split('.')[0]
        return batch_dict

@register_dataset(name='coloradar')
class ColoradarDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None, mode: str = "train"):
        super().__init__(root, transforms)
        self.root = root
        self.mode = mode
        self.data_config = {
        "train": [
            "2_22_2021_longboard_run0",
            # "2_22_2021_longboard_run1",
            # "2_22_2021_longboard_run2",
            # # "2_22_2021_longboard_run3",           #aspen
            # "2_24_2021_aspen_run0",
            # "2_24_2021_aspen_run1",
            # # "2_24_2021_aspen_run2",
            # "2_24_2021_aspen_run3",           #aspen
            # "2_28_2021_outdoors_run0",
            # "2_28_2021_outdoors_run1",
            # "2_28_2021_outdoors_run2",
            # # "2_28_2021_outdoors_run3",        #outdoor
            # "12_21_2020_arpg_lab_run0",
            # "12_21_2020_arpg_lab_run1",
            # "12_21_2020_arpg_lab_run2",
            # # "12_21_2020_arpg_lab_run3",       #arpg
            # "12_21_2020_ec_hallways_run0",
            # "12_21_2020_ec_hallways_run1",
            # "12_21_2020_ec_hallways_run2",   
            # # "12_21_2020_ec_hallways_run3", #hallway
            # "2_23_2021_edgar_classroom_run0",
            # "2_23_2021_edgar_classroom_run1",
            # "2_23_2021_edgar_classroom_run2", 
            # # "2_23_2021_edgar_classroom_run3", #classroom
            # "2_23_2021_edgar_army_run0",
            # "2_23_2021_edgar_army_run1",
            # "2_23_2021_edgar_army_run2", 
            # "2_22_2021_longboard_run3",
            # "2_22_2021_longboard_run4",
            # "2_22_2021_longboard_run5",
            # "2_22_2021_longboard_run6",
            # "2_22_2021_longboard_run7",       #longboard
            # "2_24_2021_aspen_run3",
            # "2_24_2021_aspen_run4",
            # "2_24_2021_aspen_run5",         
            # "2_24_2021_aspen_run6",
            # "2_24_2021_aspen_run7",
            # "2_24_2021_aspen_run8",   
            # # "2_24_2021_aspen_run9",
            # "2_24_2021_aspen_run10",
            # "2_24_2021_aspen_run11",          #aspen
            # "2_28_2021_outdoors_run3",
            # "2_28_2021_outdoors_run4",
            # # "2_28_2021_outdoors_run5",        
            # "2_28_2021_outdoors_run6",
            # "2_28_2021_outdoors_run7",
            # "2_28_2021_outdoors_run8",    
            # # "2_28_2021_outdoors_run9",        #outdoor
            # "12_21_2020_arpg_lab_run3", 
            # "12_21_2020_arpg_lab_run4",       #arpg
            # "12_21_2020_ec_hallways_run3",
            # "12_21_2020_ec_hallways_run4",    #hallway
            # # "2_23_2021_edgar_classroom_run3", 
            # "2_23_2021_edgar_classroom_run4", 
            # "2_23_2021_edgar_classroom_run5", #classroom
            # # "2_23_2021_edgar_army_run3",
            # # "2_23_2021_edgar_army_run4",
            # "2_23_2021_edgar_army_run5",      #army
        ],
        "test": [
            "2_22_2021_longboard_run3",
            # "2_22_2021_longboard_run4",
            # "2_22_2021_longboard_run5",
            # "2_22_2021_longboard_run6",
            # "2_22_2021_longboard_run7",       #longboard
        ] 
        }
        self.Radar, self.Lidar, self.Name = self._init_dataset()

    def _load_data_coloradar(self, radarpath, lidarpath, seqname):
        print("seqname", seqname)
        files = os.listdir(radarpath)
        files.sort()
        radar = []
        failed_index = []
        for i in files:
            path = os.path.join(radarpath, i)
            try:
                radar_img = Image.open(path).convert('L')
                radar_img = np.array(radar_img)
                radar.append(radar_img)
            except:
                print('can not read: '+path)
                failed_index.append(i)
                continue

        files = os.listdir(lidarpath)
        files.sort()
        lidar = []
        for i in files:
            if i in failed_index:
                continue
            path = os.path.join(lidarpath, i)
            try:
                lidar_img = Image.open(path).convert('L')
                lidar_img = np.array(lidar_img)
                lidar.append(lidar_img)
            except:
                print('can not read: '+path)
                failed_index.append(i)
                continue

        name_list = []
        for i in files:
            if i not in failed_index:
                name_list.append(seqname + "_{:06d}".format(int(i.split('.')[0])))

        if len(radar) == len(lidar) == len(name_list):
            return radar, lidar, name_list
        else:
            import pdb; pdb.set_trace()

    def _init_dataset(self):
        Radar, Lidar, Name = [], [], []
        for i in self.data_config[self.mode]:
            try:
                radar, lidar, name = self._load_data_coloradar(
                    os.path.join(self.root, i, "range_azimuth_heatmap"),
                    os.path.join(self.root, i, "lidar_pcl_bev_polar_img"),
                    i
                )
                Radar += radar
                Lidar += lidar
                Name += name
            except:
                import pdb; pdb.set_trace()
        print(f"Using {self.data_config[self.mode]} for {self.mode}")
        print(f"{self.mode.capitalize()} data - {len(Radar)}")
        return Radar, Lidar, Name

    def __len__(self):
        return len(self.Radar)

    def __getitem__(self, index):
        radar_data = torch.tensor(self.Radar[index]).unsqueeze(0)
        lidar_data = torch.tensor(self.Lidar[index]).unsqueeze(0)

        radar_data = torch.cat([radar_data, radar_data, radar_data], dim=0)
        lidar_data = torch.cat([lidar_data, lidar_data, lidar_data], dim=0)

        batch_dict = {
            'radar_ra': radar_data,
            'lidar_ra': lidar_data,
            'data_fname': self.Name[index]
        }
        
        if self.transforms:
            batch_dict = self.transforms(batch_dict)

        return batch_dict


@register_dataset(name='kradar')
class datasetKradar(torch.utils.data.Dataset):
    def __init__(self, root):
        super().__init__()
        self.infos = self.readtxt(root)
        self.data_shape = [256, 32, 128]
        self.lidar_data_list = []
        self.radar_data_list = []
        print('loading data...')
        for info in self.infos:
            lidar_fname, radar_fname = info['lidar_rea_fname'], info['radar_rea_fname']
            lidar_data = np.fromfile(lidar_fname, dtype=np.float32).reshape(self.data_shape).mean(axis=1).squeeze()
            radar_data = np.fromfile(radar_fname, dtype=np.float32).reshape(self.data_shape).mean(axis=1).squeeze()

            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(lidar_data)
            # plt.subplot(1, 2, 2)
            # plt.imshow(radar_data)
            # plt.savefig(f'test_kradar.png')
            # plt.close()

            self.lidar_data_list.append(lidar_data)
            self.radar_data_list.append(radar_data)
        print('data loaded!')

    def __len__(self,):
        return len(self.infos)

    def __getitem__(self, idx):
        return self.getdata(idx)

    def getdata(self, idx):
        data_dict = {}
        lidar = torch.tensor(self.lidar_data_list[idx]/self.lidar_data_list[idx].max()).unsqueeze(0)
        # radar = torch.tensor(self.radar_data_list[idx]/self.radar_data_list[idx].max()).unsqueeze(0)
        radar = torch.pow(2**torch.tensor(self.radar_data_list[idx]), 0.5).unsqueeze(0)
        radar = radar/radar.max()
        data_dict['lidar_ra'] = torch.cat([lidar, lidar, lidar], dim=0)
        data_dict['radar_ra'] = torch.cat([radar, radar, radar], dim=0)
        radar_fname = 'radar_' + self.infos[idx]['radar_rea_fname'].split('/')[-3] + '-' + self.infos[idx]['radar_rea_fname'].split('/')[-1].split('.')[0]
        lidar_fname = 'lidar_' + self.infos[idx]['lidar_rea_fname'].split('/')[-3] + '-' + self.infos[idx]['lidar_rea_fname'].split('/')[-1].split('.')[0]
        data_dict['data_fname'] = radar_fname + '-' + lidar_fname
        return data_dict

    def readtxt(self, fname):
        index_list = []
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                index_info = {}
                index_info['radar_rea_fname'], index_info['lidar_rea_fname'] = line.strip().split(' ')
                index_list.append(index_info)
        return index_list