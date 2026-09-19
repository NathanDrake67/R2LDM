import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from det3d.datasets.pipelines.compose import S2D_Compose

from .registry import DATASETS
from .pipelines import Compose


@DATASETS.register_module
class PointCloudDataset(Dataset):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    NumPointFeatures = -1
    CLASSES = None

    def __init__(
        self,
        root_path,
        info_path,
        pipeline=None,
        test_mode=False,
        class_names=None,
        **kwrags
    ):
        self._info_path = info_path
        self._root_path = Path(root_path)
        self._class_names = class_names

        self.test_mode = test_mode

        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def __getitem__(self, index):
        """This function is used for preprocess.
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            if training:
                labels
                reg_targets
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata, in kitti, image index is saved in metadata
        }
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        """Dataset must provide a unified function to get data.
        Args:
            query: int or dict. this param must support int for training.
                if dict, should have this format (no example yet):
                {
                    sensor_name: {
                        sensor_meta
                    }
                }
                if int, will return all sensor data.
                (TODO: how to deal with unsynchronized data?)
        Returns:
            sensor_data: dict.
            if query is int (return all), return a dict with all sensors:
            {
                sensor_name: sensor_data
                ...
                metadata: ... (for kitti, contains image_idx)
            }

            if sensor is lidar (all lidar point cloud must be concatenated to one array):
            e.g. If your dataset have two lidar sensor, you need to return a single dict:
            {
                "lidar": {
                    "points": ...
                    ...
                }
            }
            sensor_data: {
                points: [N, 3+]
                [optional]annotations: {
                    "boxes": [N, 7] locs, dims, yaw, in lidar coord system. must tested
                        in provided visualization tools such as second.utils.simplevis
                        or web tool.
                    "names": array of string.
                }
            }
            if sensor is camera (not used yet):
            sensor_data: {
                data: image string (array is too large)
                [optional]annotations: {
                    "boxes": [N, 4] 2d bbox
                    "names": array of string.
                }
            }
            metadata: {
                image_idx: ...
            }
            [optional]calib # only used for kitti
        """
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        """
        If you want to eval by my KITTI eval function, you must
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        raise NotImplementedError

    def pre_pipeline(self, results):
        results["img_prefix"] = self.img_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results["bbox_fields"] = []
        results["mask_fields"] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def prepare_train_input(self, idx):
        raise NotImplementedError

    def prepare_test_input(self, idx):
        raise NotImplementedError

@DATASETS.register_module
class S2DDataset(Dataset):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    CLASSES = None

    def __init__(
        self,
        pipeline=None,
    ):
        self._set_group_flag()
        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = S2D_Compose(pipeline)

    def __getitem__(self, index):
        """
        
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_points_path(self, folder, frame_index, dtype: str, ext='.bin'):
        """Get the points for the given frame
        params:
        folder: str
            The folder name
        frame_index: int
            The frame index
        dtype: str
            The type of data to load:Lidar or Radar
        ext: str
            The extension of the file
        returns:
        points: np.ndarray
            The points for the frame
        """
        f_str = "{:05d}{}".format(frame_index, ext)
        points_path = os.path.join(
            self.root_path, folder, dtype, f_str)
        return points_path
    
    def get_image_path(self, folder, frame_index, sub_folder='camera', ext='.jpg'):
        f_str = "{:05d}{}".format(frame_index, ext)
        image_path = os.path.join(
            self.root_path, folder, sub_folder, f_str)
        return image_path
    
    def get_student_path(self, folder, frame_index, sub_folder='student_voxel', ext='.jpg'):
        f_str = "{:05d}{}".format(frame_index, ext)
        image_path = os.path.join(
            self.root_path, folder, sub_folder, f_str)
        return image_path
    
    def get_teacher_path(self, folder, frame_index, sub_folder='teacher_voxel', ext='.jpg'):
        f_str = "{:05d}{}".format(frame_index, ext)
        image_path = os.path.join(
            self.root_path, folder, sub_folder, f_str)
        return image_path  

    def downsample_points(self, points, num_points):
        """Downsample the number of points to num_points
        """
        if len(points) > num_points:
            choice = np.random.choice(len(points), num_points, replace=False)
            points = points[choice]
        return points

    def lidar_preprocess(self, points):
        """Restrict LiDAR points to the configured field of view."""
        # FOV
        ang = np.arctan2(points[:, 1], points[:, 0])
        ang_deg = np.degrees(ang)
        mask = (ang_deg>=-60) and (ang_deg<=60)
        fillter_Points = points[mask]
        return fillter_Points   
    
    def radar_preprocess(self, points, h_thresh=-1.6, deg_thresh=60):
        """Restrict radar points by field of view and height."""
        # FOV
        ang = np.arctan2(points[:, 1], points[:, 0])
        ang_deg = np.degrees(ang)
        mask_fov = (ang_deg>=-deg_thresh) and (ang_deg<=deg_thresh)
        # range
        mask_range = points[:, 2] >= h_thresh
        mask = mask_fov and mask_range
        fillter_Points = points[mask]
        return fillter_Points

    def cv2_loader(self, path, color_type='RGB'):
        img = cv2.imread(path)
        if color_type == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def readlines(self, filename):
        """Read the lines of the split file
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)
