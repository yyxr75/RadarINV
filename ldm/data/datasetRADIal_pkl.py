# Creates a dataloader in batches for training and testing

import numpy as np
import torch
import pickle

# dataset_rootPath = '/scratch/user/yanlongyang/Project1_Diffusion_related/dataset_with_labels/'
# dataset_rootPath = '/data/yyl-fusion-demo/dataset_with_labels/'
# dataset_rootPath = '/root/autodl-tmp/dataset_with_labels/'
dataset_rootPath = '/scratch/project/cpautodriving/yanlongyang/dataset_with_labels/'

class DatasetRADIal(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_root, 
        RBINS=512,
        ABINS=768, 
        M=0,
    ):
        with open(data_root, 'rb') as f:
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
        print('loading dataset...')
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
        print('dataset loaded!')

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
        # ----------------------
        # 读取目标标注文件，按照帧分开
        # ----------------------
        # self.targets_infos = []
        # if target_filename is not None:
        #     # 使用 pandas 读取数据
        #     import pandas as pd
        #     df = pd.read_csv(target_filename, skipinitialspace=True)
        #     # 按照 'numSample' 列分组，并将每个组转换为一个数据框，然后将这些数据框放入一个列表中, by GPT
        #     # 0 numSample	1 x1_pix	2 y1_pix	3 x2_pix	4 y2_pix	5 laser_X_m	6 laser_Y_m	7 radar_X_m	
        #     # 8 radar_Y_m	9 radar_R_m	10 radar_A_deg	11 radar_D_mps	12 radar_P_db	
        #     # 13 dataset	14 index	15 Annotation   16 Difficult
        #     self.targets_infos = [group.values[:,:13].astype(np.float) for _, group in df.groupby('numSample')]


    def __len__(self):
        return len(self.input_data)

    def __filenames__(self):
        return [x[-36:-17]+x[-10:-4] for x in self.labels]

    def get_lidar(self, label_filename):

        # a = Image.open(label_filename)
        parts = label_filename.split('/')
        parts[-2] = 'lidars_mask'
        label_filename = '/'.join(parts)
        a = np.fromfile(label_filename)
        y = torch.Tensor(np.reshape(a, (1,self.RBINS,self.ABINS)))
        return y

    def get_radar(self, input_filename):
        # a = Image.open(input_filename)
        parts = input_filename.split('/')
        parts[-2] = 'radars_ra_interp'
        input_filename = '/'.join(parts)
        a = np.fromfile(input_filename)
        a = np.interp(np.linspace(0, len(a)-1, self.RBINS*self.ABINS), np.arange(len(a)), a)
        a = a.reshape(1, self.RBINS, self.ABINS)
        ra = (a-a.min())/(a.max()-a.min())
        X = torch.Tensor(ra)
        return X

    def __getitem__(self, index):
        # Select sample
        if self.history == 0:
            input_filename = self.input_data[index]
            label_filename = self.labels[index]
            X, y = self.get_radar(input_filename), self.get_lidar(label_filename)
        else:
            X = torch.Tensor([])
            input_filenames = self.input_data[index]
            label_filename = self.labels[index]
            for i in input_filenames:
                xx = self.get_radar(i)
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
