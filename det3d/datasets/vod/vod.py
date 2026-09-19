import os
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import S2DDataset
from torch.utils.data import Dataset

from det3d.datasets.registry import DATASETS
# from det3d.datasets.s2d_utils import load_lidar_bin, load_radar_bin


@DATASETS.register_module
class VodDataset(S2DDataset):
    def __init__(
        self,
        root_path,
        pipeline,
        filenames,
        nsweeps=1,
        is_train=False,
        is_test=False,
        two_stage=False,
        all_feature=False,
        only_vr=False,
        only_rcs=False,
        num_features=3,
        lidar_input="lidar_seq_preprocess",
        lidar_gt="lidar_seq_preprocess",
        radar_1frame=True,
        **kwargs,
    ):
        self.filenames = self.readlines(filenames)
        self.root_path = root_path
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))
        super(VodDataset, self).__init__(pipeline=pipeline)
        self.is_train = is_train
        self.is_test = is_test
        self.two_stage = two_stage
        self.lidar_input = lidar_input
        self.lidar_gt = lidar_gt
        self.all_feature = all_feature
        self.only_vr = only_vr
        self.only_rcs = only_rcs
        self.num_features = num_features
        self.train_list = os.listdir(os.path.join(self.root_path, "train"))
        self.val_list = os.listdir(os.path.join(self.root_path, "val"))
        self.test_list = os.listdir(os.path.join(self.root_path, "test"))
        self.test_list.extend(os.listdir(os.path.join(self.root_path, "test_rest")))
        self.radar_1frame = radar_1frame

    def index_to_folder_and_frame_idx(self, index, is_train, is_test):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split("/")
        folder = line[0]
        frame_index = int(line[1])

        if folder in self.val_list:
            folder = "val/" + folder
        elif folder in self.test_list:
            if folder == '24':
                folder = "test/" + folder
            else:
                folder = "test_rest/" + folder
        else:
            folder = "train/" + folder

        return folder, frame_index
    
    def __getitem__(self, index):
        res_dict = {
            'lidar':{
                'points':None,
            },
            'lidar_gt':{
                'points':None,
            },
            'radar':{
                'points':None,
            },
            'image': None, 
            'folder': None,
            'frame_idx': None,
        }

        folder, frame_index = self.index_to_folder_and_frame_idx(index, self.is_train, self.is_test)
        
        # get the lidar points
        lidar_path = self.get_points_path(folder, frame_index, self.lidar_input)
        lidar_points = load_lidar_bin(lidar_path, self.num_features)
        lidar_gt_path = self.get_points_path(folder, frame_index, self.lidar_gt)
        lidar_gt_points = load_lidar_bin(lidar_gt_path, self.num_features)

        res_dict['lidar']['points'] = lidar_points
        res_dict['lidar_gt']['points'] = lidar_gt_points

        if self.radar_1frame:
            radar_input = "radar"
        else:
            radar_input = "radar_" + str(self.nsweeps) + "frames_seq_preprocess"
        radar_path = self.get_points_path(folder, frame_index, radar_input)
        radar_points, rcs, vr = load_radar_bin(
            radar_path,
            all=self.all_feature,
            only_vr=self.only_vr,
            only_rcs=self.only_rcs,
            num_features=self.num_features,
        )
        if self.all_feature:
            if self.only_rcs:
                radar_points = np.concatenate((radar_points, rcs[:, np.newaxis]), axis=1)
            if self.only_vr:
                radar_points = np.concatenate((radar_points, vr[:, np.newaxis]), axis=1)
        res_dict['radar']['points'] = radar_points

        if self.is_test:
            image_path = self.get_image_path(folder, frame_index)
            image = self.cv2_loader(image_path)
            res_dict['image'] = image
            res_dict['folder'] = folder
            res_dict['frame_idx'] = frame_index

        data = self.pipeline(res_dict)
        return data

    def __len__(self):
        return len(self.filenames)

def load_lidar_bin(file_path, num_features=4):
    """Load a LiDAR point cloud from a binary file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :num_features]

def load_radar_bin(file_path, all=True, only_vr=False, only_rcs=False, num_features=4):
    """Load a radar point cloud from a binary file."""
    radar_scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 7))
    if num_features > 3:
        if all:
            points = radar_scan[:, :6]
        elif only_vr:
            points = radar_scan[:, [0, 1, 2, 4]] #(N, 4)
        elif only_rcs:
            points = radar_scan[:, [0, 1, 2, 3]] #(N, 4)
    elif num_features == 3:
        points = radar_scan[:, :3]
    
    rcs = radar_scan[:, 3]
    vr = radar_scan[:, 4]

    return points, rcs, vr

def get_image_path(self, folder, frame_index):
    f_str = "{:05d}{}".format(frame_index, self.img_ext)
    image_path = os.path.join(
        self.data_path, folder, "camera", f_str)
    return image_path
