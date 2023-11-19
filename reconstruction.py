#reconstruction
import open3d as o3d

#读取语义分割图像
img = cv2.imread('/media/xpan/文档/Program/pycharm/3D-Reconstruction/imgs/group3_l_label.png')

f = new_in_config['group_3'][0]
cx = new_in_config['group_3'][1]
cy = new_in_config['group_3'][2]
b = new_in_config['group_3'][3]

point_cloud = o3d.geometry.PointCloud()
points = []
depth_max = []
for i in range(4620):
    for j in range(1800):
        if img[j][i][2] != 0:
#             if disp[j][i] > 50 and disp[j][i] < 320:
            disp[j][i] = disp[j][i] + 4980
            depth = (b*f)/(-disp[j][i])
            x = (i+18880-cx)*b/(-disp[j][i])
            y = (j+4440-cy)*b/(-disp[j][i])
            n = np.array([x,y,depth])
            points.append(n)

np.array(points)
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])

num_neighbors = 100 # K邻域点的个数
std_ratio = 2.0 # 标准差乘数
sor_pcd23, ind = point_cloud.remove_statistical_outlier(num_neighbors, std_ratio)
aabb23 = sor_pcd23.get_axis_aligned_bounding_box()
aabb23.color = (1, 0, 0)
print((aabb23.get_max_bound()-aabb23.get_min_bound()))

o3d.visualization.draw_geometries([sor_pcd23,aabb23])
