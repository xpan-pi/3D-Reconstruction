from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np

def pointcloud_pre():
    target_pcd = o3d.io.read_point_cloud('./output_model.pcd')
    o3d.visualization.draw_geometries([target_pcd])

    # 设置半径范围
    radius = 0.01
    # 获取点的坐标
    points = np.asarray(target_pcd.points)

    # 创建一个KDTree
    kdtree = o3d.geometry.KDTreeFlann(target_pcd)
    out_point_index = []
    for i, point in enumerate(points):
    # 查询在半径范围内的点的索引
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if k<20:
            out_point_index.append(i)
            # print("{}周围点的数量2:".format(i+1), k)
    after_points = np.delete(target_pcd.points,out_point_index, axis=0)

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(after_points)
    out_pcd.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([out_pcd])

    # 转换点云数据为numpy数组
    points = np.asarray(out_pcd.points)
    # 选择适当的距离阈值用于去除离群点
    distance_threshold = 0.01
    # 使用最近邻算法计算每个点的K个最近邻点
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    # 计算每个点的平均距离
    avg_distances = np.mean(distances, axis=1)
    # 根据阈值筛选离群点的索引
    outlier_indices = np.where(avg_distances > distance_threshold)[0]
    # 剔除离群点
    filtered_points = np.delete(points, outlier_indices, axis=0)
    # 创建新的点云模型对象并设置剔除离群点后的点的坐标
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.visualization.draw_geometries([new_cloud])
    o3d.io.write_point_cloud("./67new_cloud.pcd",new_cloud)

if __name__ == '__main__':
    pointcloud_pre()
