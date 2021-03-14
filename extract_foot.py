import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
import copy
import cv2

sns.set()


class ExtractFoot():
    def plane_segmentation(self, pcd):
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        return inlier_cloud, outlier_cloud

    def DBSCAN_clustering(self, pcd):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(
            labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd, labels

    def select_max_class(self, labels):
        classes = []
        counts = []
        for i in range(max(labels)):
            counts.append(np.count_nonzero(labels == i))
        return np.argmax(counts)

    def pcd_select_by_label(self, pcd, labels, targets, invert=False):
        indexs = np.array([], dtype=np.int)
        for target in targets:
            indexs_part = np.where(labels == target)[0]
            indexs = np.concatenate([indexs, indexs_part])
        return pcd.select_by_index(indexs, invert=invert)

    def bounding_volumes(self, pcd):
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        return aabb

    def extract_foot(self, pcd):
        pcd_plane, pcd_no_plane = self.plane_segmentation(pcd)
        pcd_clastering, labels = self.DBSCAN_clustering(
            copy.deepcopy(pcd_no_plane))
        classes = self.select_max_class(labels)
        pcd_part = self.pcd_select_by_label(pcd_no_plane, labels, [classes])

        aabb = self.bounding_volumes(pcd_part)

        bounding_points = np.array(aabb.get_box_points())
        pcd_part = copy.deepcopy(pcd_part).translate(
            (-max(bounding_points[:, 0]), -max(bounding_points[:, 1]), -max(bounding_points[:, 2])))
        return pcd_part


def main():
    color_raw = o3d.io.read_image('color.png')
    depth_raw = o3d.io.read_image('depth.png')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    pcd_row = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd = ExtractFoot().extract_foot(pcd_row)
    o3d.io.write_point_cloud('pcd.pcd', pcd)
    o3d.visualization.draw_geometries([pcd_row])
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
