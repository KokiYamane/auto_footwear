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


class Curve():
    def __init__(self, point0, pointf, v0, vf, a0=0, af=0):
        t0 = point0[0]
        tf = pointf[0]
        x0 = point0[1]
        xf = pointf[1]

        dt = tf - t0
        self.t0 = t0
        self.a = np.zeros(6)
        self.a[0] = x0
        self.a[1] = v0
        self.a[2] = a0 / 2
        self.a[3] = (20 * (xf - x0) - (8 * vf + 12 * v0) *
                     dt - (3 * a0 - af) * dt**2) / (2 * dt**3)
        self.a[4] = (30 * (x0 - xf) + (14 * vf + 16 * v0) *
                     dt + (3 * a0 - 2 * af) * dt**2) / (2 * dt**4)
        self.a[5] = (12 * (xf - x0) - 6 * (vf + v0) *
                     dt - (a0 - af) * dt**2) / (2 * dt**5)

    def calc(self, t_row):
        t = t_row - self.t0
        return self.a[0] + self.a[1] * t + self.a[2] * t**2 + \
            self.a[3] * t**3 + self.a[4] * t**4 + self.a[5] * t**5


def fit_carve(pcd, pos, range=0.5, axis=0):
    points = np.array(pcd.points) * -1000
    points_part = points[(pos - range < points[:, axis])
                         & (points[:, axis] < pos + range)]
    if len(points_part) == 0:
        return [], []

    # points_part = np.insert(
    #     points_part, 0, [[0, max(points_part[:, 1]), 0]], axis=0)
    # points_part = np.append(
    #     points_part, [[0, min(points_part[:, 1]), 0]], axis=0)

    def cf(x, y): return np.polyfit(x, y, 3)
    fitted_curve = np.poly1d(
        cf(points_part[:, 1], points_part[:, 2]))(points_part[:, 1])
    return points_part, fitted_curve


def calc_fitted_curve(x, y):
    return np.poly1d(np.polyfit(x, y, 4))(x)


def make_mesh(curve1, curve2):
    length = min(len(curve1), len(curve2))
    obj = mesh.Mesh(np.zeros(length * 2, dtype=mesh.Mesh.dtype))
    for i in range(length - 1):
        obj.vectors[i * 2] = [curve1[i], curve1[i + 1], curve2[i]]
        obj.vectors[i * 2 + 1] = [curve2[i], curve1[i + 1], curve2[i + 1]]
    obj.remove_duplicate_polygons = True
    return obj


def main():
    # color_raw = o3d.io.read_image('color.png')
    # depth_raw = o3d.io.read_image('depth.png')
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     color_raw, depth_raw, convert_rgb_to_intensity=False)
    # pcd_row = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_image,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # pcd = ExtractFoot().extract_foot(pcd_row)
    pcd = o3d.io.read_point_cloud('pcd.pcd')
    # o3d.visualization.draw_geometries([pcd])

    fitted_curve_list = []
    outline_points1 = []
    outline_points2 = []
    for i in range(0, 180, 1):
        points_part, fitted_curve = fit_carve(pcd, i)
        # plt.scatter(points_part[:, 1], points_part[:, 2])
        # plt.plot(points_part[:, 1], fitted_curve + 3, c="red", label="fitted")
        # plt.show()
        if len(points_part) == 0:
            continue
        carve_points = np.stack(
            [[i] * len(fitted_curve), points_part[:, 1], fitted_curve + 3])
        carve_points = carve_points.transpose()
        fitted_curve_list.append(carve_points)
        outline_points1.append(points_part[-1])
        outline_points2.append(points_part[0])

    obj_list = []
    for i in range(1, len(fitted_curve_list) - 1):
        obj = make_mesh(fitted_curve_list[i], fitted_curve_list[i + 1])
        obj_list.append(obj.data.copy())
    obj = mesh.Mesh(np.concatenate(obj_list))
    obj.save('sample.stl')

    mesh_data = o3d.io.read_triangle_mesh('sample.stl')
    # o3d.visualization.draw_geometries([mesh_data])

    points = np.array(pcd.points) * -1000
    # points_part = points[(39 < points[:, 1]) & (points[:, 1] < 40)]
    points_part = points
    sort_index = np.argsort(points_part[:, 0])
    points_part = points_part[sort_index]
    # points_part = np.insert(
    #     points_part, 0, [[0, max(points_part[:, 1]), 0]], axis=0)
    # points_part = np.append(
    #     points_part, [[0, min(points_part[:, 1]), 0]], axis=0)
    print(points_part)
    # def cf(x, y): return np.polyfit(x, y, 4)
    # fitted_curve = np.poly1d(
    #     cf(points_part[:, 0], points_part[:, 1]))(points_part[:, 0])
    # plt.scatter(points_part[:, 0], points_part[:, 1])
    outline_points1 = np.array(outline_points1)
    outline_points2 = np.array(outline_points2)
    # def cf(x, y): return np.polyfit(x, y, 4)
    # fitted_curve = np.poly1d(
    #     cf(outline_points[:, 0], outline_points[:, 1]))(outline_points[:, 0])
    # x = outline_points[:, 0]
    # y = outline_points[:, 1]
    fitted_curve_points1 = calc_fitted_curve(
        outline_points1[:, 0], outline_points1[:, 1]) - 3
    fitted_curve_points2 = calc_fitted_curve(
        outline_points2[:, 0], outline_points2[:, 1]) + 3

    def diff_0(x, y):
        return (y[1] - y[0]) / (x[1] - x[0])

    def diff_f(x, y):
        return (y[-1] - y[-2]) / (x[-1] - x[-2])
    toes_curve = Curve([fitted_curve_points1[0], outline_points1[0, 0]],
                       [fitted_curve_points2[0], outline_points2[0, 0]],
                       diff_0(fitted_curve_points1, outline_points1[:, 0]),
                       diff_0(fitted_curve_points2, outline_points2[:, 0]))
    heels_curve1 = Curve([fitted_curve_points1[-1], outline_points1[-1, 0]],
                         #  [40, 250],
                         [fitted_curve_points2[-1], outline_points2[-1, 0]],
                         diff_f(fitted_curve_points1, outline_points1[:, 0]),
                         diff_f(fitted_curve_points2, outline_points2[:, 0]),
                         #  0,
                         )
    heels_curve2 = Curve([fitted_curve_points2[-1], outline_points2[-1, 0]],
                         [40, 250],
                         diff_f(fitted_curve_points2, outline_points2[:, 0]),
                         0)

    plt.scatter(outline_points1[:, 0], outline_points1[:, 1])
    plt.scatter(outline_points2[:, 0], outline_points2[:, 1])
    plt.plot(outline_points1[:, 0], fitted_curve_points1, c='red')
    plt.plot(outline_points2[:, 0], fitted_curve_points2, c='red')
    plt.scatter([250], [40])
    x = np.arange(fitted_curve_points1[0], fitted_curve_points2[0], 0.1)
    plt.plot(toes_curve.calc(x), x, c='red')
    x = np.arange(fitted_curve_points1[-1], 40, 0.1)
    plt.plot(heels_curve1.calc(x), x, c='red')
    x = np.arange(40, fitted_curve_points2[-1], 0.1)
    plt.plot(heels_curve2.calc(x), x, c='red')
    plt.axes().set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()
