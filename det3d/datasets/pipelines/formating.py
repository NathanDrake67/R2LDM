from ..registry import PIPELINES


class DataBundle:
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat:
    def __init__(self, double_flip: bool = False, **kwargs):
        self.double_flip = double_flip

    def __call__(self, res):
        lidar = res["lidar"]
        voxels = lidar["voxels"]
        dense_voxels = lidar["dense_voxels"]
        reconstruction_voxels = lidar["reconstruction_voxels"]
        reconstruction_voxels_2 = lidar["reconstruction_voxels_2"]
        reconstruction_voxels_4 = lidar["reconstruction_voxels_4"]

        return {
            "metadata": res["metadata"],
            "points": lidar["points"],
            "dense_points": lidar["dense_points"],
            "voxels": voxels["voxels"],
            "dense_voxels": dense_voxels["voxels"],
            "shape": voxels["shape"],
            "num_points": voxels["num_points"],
            "dense_num_points": dense_voxels["num_points"],
            "num_voxels": voxels["num_voxels"],
            "dense_num_voxels": dense_voxels["num_voxels"],
            "coordinates": voxels["coordinates"],
            "dense_coordinates": dense_voxels["coordinates"],
            "reconstruction_points": lidar["reconstruction_points"],
            "reconstruction_voxels": reconstruction_voxels["voxels"],
            "reconstruction_coordinates": reconstruction_voxels["coordinates"],
            "reconstruction_num_voxels": reconstruction_voxels["num_voxels"],
            "reconstruction_num_points": reconstruction_voxels["num_points"],
            "reconstruction_voxels_2": reconstruction_voxels_2["voxels"],
            "reconstruction_coordinates_2": reconstruction_voxels_2["coordinates"],
            "reconstruction_num_voxels_2": reconstruction_voxels_2["num_voxels"],
            "reconstruction_num_points_2": reconstruction_voxels_2["num_points"],
            "reconstruction_voxels_4": reconstruction_voxels_4["voxels"],
            "reconstruction_coordinates_4": reconstruction_voxels_4["coordinates"],
            "reconstruction_num_voxels_4": reconstruction_voxels_4["num_voxels"],
            "reconstruction_num_points_4": reconstruction_voxels_4["num_points"],
        }
