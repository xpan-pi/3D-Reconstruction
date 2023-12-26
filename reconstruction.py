import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import argparse
import os
import sys
sys.path.append('/home/xpan/Desktop/program/SoftRas-master/examples/recon')

import imageio
import torch
import tqdm

import soft_renderer.functional as srf
try:
    from softras import models, models_large
    from softras.utils import img_cvt
except ImportError:
    import models, models_large
    from utils import img_cvt

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--softras-dir', type=str, default='/home/xpan/Desktop/program/SoftRas-master')
parser.add_argument('-d', '--model-path', type=str, default='/home/xpan/Desktop/program/SoftRas-master/data/results/models/record_64_1500/checkpoint_0010000.pth.tar')
parser.add_argument('-is', '--image-size', type=int, default=64)
parser.add_argument('-sv', '--sigma-val', type=float, default=0.01)
parser.add_argument('--shading-model', action='store_true', help='test shading model')
parser.add_argument('-img', '--image-path', type=str, default='/home/xpan/Desktop/program/SoftRas-master/big_obj.png')
args = parser.parse_args(args=[])
print(f'load: {args.model_path}')
obj_path=f'{args.softras_dir}/data/obj/sphere/sphere_642.obj'
if args.shading_model:
    model = models_large.Model(obj_path, args=args)
else:
    model = models.Model(obj_path, args=args)

model = model.cuda()
state_dicts = torch.load(args.model_path)

model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

def mesh_recon(x):
    if x.ndim == 4:
        pass
    elif x.ndim == 3:
        x = x[np.newaxis, :]
    else:
        raise ValueError('X must be 3/4 dims: CHW/BCHW.')

    assert x.shape[1] == 4, 'C must be 4: rgba.'
    # print(f'  {x.shape}')

    x = np.ascontiguousarray(x)
    x = x.astype('float32') / 255.
    x = torch.autograd.Variable(torch.from_numpy(x)).cuda()
    _, vertices, faces = model(x, task='test')

    return x, vertices, faces

def create_model():
    if args.image_path and os.path.exists(args.image_path):
        print(f'read: {args.image_path}')
        image = imageio.imread(args.image_path)
        x = image.transpose(2, 0, 1) # HWC > CHW

        _, vertices, faces = mesh_recon(x)

        obj_path = os.path.splitext(args.image_path)[0] + '.obj'
        print(f'save: {obj_path}')
        srf.save_obj(obj_path, vertices[0], faces[0])

def ellipse_residuals(params, x, y):
    # 椭圆方程参数
    h, k, a, b = params
    # 计算椭圆方程
    ellipse_eq = ((x - h) / a) ** 2 + ((y - k) / b) ** 2 - 1
    return ellipse_eq

def fit_ellipse(x, y):
    # 初始化参数估计值
    h_guess = np.mean(x)
    k_guess = np.mean(y)
    a_guess = np.max(x) - np.min(x)
    b_guess = np.max(y) - np.min(y)
    params_guess = [h_guess, k_guess, a_guess, b_guess]
    # 最小二乘拟合
    result = least_squares(ellipse_residuals, params_guess, args=(x, y))
    # 获取拟合得到的参数
    params = result.x
    return params

def create_points(mesh, n_pts, ratio):
    pcd = mesh.sample_points_uniformly(n_pts)

    points = np.asarray(pcd.points)
    scaled_points = points * ratio
    model_ratio = o3d.geometry.PointCloud()
    model_ratio.points = o3d.utility.Vector3dVector(scaled_points)

    return model_ratio

def create_z_fit_points(pcd, num_r, num_points):
    point_cloud = o3d.geometry.PointCloud()
    num_x = []
    num_y = []
    num_z = []
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    dis = aabb.get_max_bound() - aabb.get_min_bound()

    for k in range(0, num_r):
        b1 = [aabb.get_min_bound()[0], aabb.get_min_bound()[1], -dis[2] * 0.5 + (dis[2] * k) / num_r]
        b2 = [aabb.get_max_bound()[0], aabb.get_max_bound()[1], -dis[2] * 0.5 + (dis[2] * (k + 1)) / num_r]
        bbox1 = o3d.geometry.AxisAlignedBoundingBox(b1, b2)
        bbox1.color = (1, 0, 0)
        n_pcd = pcd.crop(bbox1)

        x = np.array(n_pcd.points)[:, 0]
        y = np.array(n_pcd.points)[:, 1]
        params_fit = fit_ellipse(x, y)
        h_fit, k_fit, a_fit, b_fit = params_fit
        theta = np.linspace(0, 2 * np.pi, num_points)
        ellipse_x = h_fit + a_fit * np.cos(theta)
        ellipse_y = k_fit + b_fit * np.sin(theta)
        if max(ellipse_y) > 2 * dis[0] or min(ellipse_y) < -2 * dis[0]:
            continue

        num_x.append(ellipse_x)
        num_y.append(ellipse_y)
        num_z.append(dis[2] * (-0.5 + k / num_r))
    tx = np.array(num_x).reshape(-1)
    ty = np.array(num_y).reshape(-1)
    tz = np.array(num_z).reshape(-1)
    n = [[tx[i], ty[i], tz[i // num_points]] for i in range(tx.shape[0])]
    point_cloud.points = o3d.utility.Vector3dVector(n)
    aabb = point_cloud.get_axis_aligned_bounding_box()
    dis = aabb.get_max_bound() - aabb.get_min_bound()
    return point_cloud

def create_last_points(pcd, n_rounds):
    point_cloud = o3d.geometry.PointCloud()

    angle_increment = 360 / n_rounds  # 每次转的角度
    aabb = pcd.get_axis_aligned_bounding_box()
    width = (aabb.get_max_bound() - aabb.get_min_bound())[0]
    height = (aabb.get_max_bound() - aabb.get_min_bound())[1]

    #     r = (aabb.get_max_bound()-aabb.get_min_bound())[1]/2
    r = (width + height) / 4
    each_h = 2 * np.pi * r * angle_increment / 360

    half_diagonal = np.sqrt(width ** 2 + height ** 2) / 2

    R = np.array([[np.cos(np.radians(angle_increment)), -np.sin(np.radians(angle_increment)), 0, 0],
                  [np.sin(np.radians(angle_increment)), np.cos(np.radians(angle_increment)), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    for k in range(n_rounds // 2):  # int(n_rounds/2)
        ##优化
        x_fit = []
        z_fit = []
        y_fit = []
        b1 = [-half_diagonal, -0.5 * each_h, aabb.get_min_bound()[2]]
        b2 = [half_diagonal, 0.5 * each_h, aabb.get_max_bound()[2]]
        bbox1 = o3d.geometry.AxisAlignedBoundingBox(b1, b2)
        bbox1.color = (1, 0, 0)
        n_pcd = pcd.crop(bbox1)

        condition = np.array(n_pcd.points)[:, 0] < 0
        z = np.array(n_pcd.points)[np.where(condition)][:, 0]
        x = np.array(n_pcd.points)[np.where(condition)][:, 2]
        y = np.array(n_pcd.points)[np.where(condition)][:, 1]
        linear_model = np.polyfit(x, z, 3)  # 三次拟合
        linear_model_fn = np.poly1d(linear_model)  # 得到拟合函数
        x_fit.append(linear_model_fn(x))
        z_fit.append(x)
        y_fit.append(y)

        condition = np.array(n_pcd.points)[:, 0] > 0
        z = np.array(n_pcd.points)[np.where(condition)][:, 0]
        x = np.array(n_pcd.points)[np.where(condition)][:, 2]
        y = np.array(n_pcd.points)[np.where(condition)][:, 1]
        linear_model = np.polyfit(x, z, 3)  # 三次拟合
        linear_model_fn = np.poly1d(linear_model)  # 得到拟合函数
        x_fit.append(linear_model_fn(x))
        z_fit.append(x)
        y_fit.append(y)

        tx = np.concatenate(x_fit)
        ty = np.concatenate(y_fit)  
        tz = np.concatenate(z_fit)
        new = [[tx[i], ty[i], tz[i]] for i in range(tx.shape[0])]
        or_n = np.array(point_cloud.points)

        n2 = np.vstack((or_n, new))
        point_cloud.points = o3d.utility.Vector3dVector(n2)
        #         o3d.visualization.draw_geometries([pcd,bbox1])
        #         plt.scatter(np.array(n_pcd.points)[np.where(condition)][:,2],np.array(n_pcd.points)[np.where(condition)][:,0])
        #         plt.scatter(tz,tx)
        #         plt.show()

        pcd.transform(R)
        point_cloud.transform(R)
    return point_cloud

def point2mesh(point_cloud):
    radius1 = 2  # 搜索半径
    max_nn = 100  # 邻域内用于估算法线的最大点数
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius1, max_nn))  #
    distances = point_cloud.compute_nearest_neighbor_distance()  #
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(
        [radius, radius * 2]))  #
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return mesh

if __name__ == '__main__':
    # 参数说明
    # n_pts --> mesh采样的点数
    # ratio --> 点云缩放倍数
    # num_round --> 点云切割份数
    # each_round --> 每圈点的个数

    # mesh = o3d.io.read_triangle_mesh('/media/xpan/文档/Program/pycharm/SoftRas-master/SoftRas-master/roi_3.obj')


    create_model()

    mesh = o3d.io.read_triangle_mesh('/home/xpan/Desktop/program/SoftRas-master/big_obj.obj')
    mesh.compute_vertex_normals()

    n_pts = 10000
    ratio = 5
    pcd = create_points(mesh, n_pts, ratio) 


    num_round = 50
    each_round = 100
    point_cloud = create_z_fit_points(pcd, num_round, each_round)

    point_last_cloud = create_last_points(point_cloud, each_round)

    mesh = point2mesh(point_last_cloud)

    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([point_cloud])
    o3d.visualization.draw_geometries([point_last_cloud])
    #     o3d.io.write_point_cloud('E:/Program/pycharm/SoftRas-master/SoftRas-master/result.ply', point_last_cloud)

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
