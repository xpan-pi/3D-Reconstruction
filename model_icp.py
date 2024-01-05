'''
代码流程：
        1、先对CREStereo重建出来的点云进行预处理，包括修正坐标系以及稀疏点云
        2、通过PCA进行点云的初步匹配
        3、通过点到面进行精匹配
'''
'''
参数说明：
        1、target_pcd:CREStereo重建出来的部分点云深度模型
        2、source_pcd:SOFTRAS重建出来的工件点云
'''
### 1、过滤点云
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pyquaternion import Quaternion
import math

def downsample_point_cloud(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd.points

### 2、将修正CREStereo得到的模型坐标系
def camera_to_world(camera_points):
    T1 = np.array([[ 0.9527905,  -0.08519282,  0.29143171, 0.000000],
                [ 0.08039387,  0.99635778,  0.02842524, 0.000000],
                [-0.29279188, -0.00365397,  0.95616921, 0.000000],
                [0.000000, 0.000000, 0.000000, 1.000000]]) ##相机校正参数
    ##相机外参
    s6_T = np.array([[0.951755, -0.306826, 0.004592, 0.132921],
    [-0.082245, -0.269477, -0.959488, 1.241468],
    [0.295633, 0.912820, -0.281710, 13.442428],
    [0.000000, 0.000000, 0.000000, 1.000000]])
    N = len(camera_points)
    world_points = []
    # 将相机坐标系中的点坐标转换为齐次坐标
    camera_points_homogeneous = np.concatenate([camera_points, np.ones((N, 1))], axis=1)
    for point in camera_points_homogeneous:
        point2 = np.dot(np.linalg.inv(T1),np.array(point))
        world_point = np.dot(np.linalg.inv(s6_T),np.array(point2))
        world_point = world_point[:3]
        world_points.append(world_point)
    return world_points

### 3、根据点云中心进行移动
def apply_transformation(source_points, transformation):
    source_points_homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    transformed_points_homogeneous = np.dot(transformation, source_points_homogeneous.T).T
    transformed_points = np.divide(transformed_points_homogeneous[:, :3], transformed_points_homogeneous[:, 3].reshape(-1, 1))
    return transformed_points

### 4、利用PCA进行点云粗匹配
##计算点云的主成分向量
def calculate_pca(points_np):
    pca = PCA()
    pca.fit(points_np)
    ##获取主成分向量
    components = pca.components_
    main_component = components[0]
    return main_component

##计算两个主成分向量的夹角
def calculate_angle(vector1,vector2):
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    cos_angle = np.dot(vector1, vector2) / (length1*length2)
    angle = np.arccos(cos_angle)
    return angle

def pcd_revolve(target_pcd,trans_pcd,angle,axis,outer_box):
    if axis =='x':
        ##先绕x旋转
        R1 = np.array([[1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)]])         #x
        trans_pcd.rotate(R1,center = outer_box.get_center())
        trans_pcd_component = calculate_pca(np.array(trans_pcd.points))
        target_pcd_component = calculate_pca(np.array(target_pcd.points))
        component_angle = calculate_angle(trans_pcd_component,target_pcd_component)
        R11 = np.array([[1, 0, 0],
        [0, np.cos(-angle), np.sin(-angle)],
        [0, -np.sin(-angle), np.cos(-angle)]]) 
        trans_pcd.rotate(R11,center = outer_box.get_center())
        return [component_angle, R1, axis]
    
    elif axis =='y':
        R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])       #y
        trans_pcd.rotate(R2,center = outer_box.get_center())
        trans_pcd_component = calculate_pca(np.array(trans_pcd.points))
        target_pcd_component = calculate_pca(np.array(target_pcd.points))
        component_angle = calculate_angle(trans_pcd_component,target_pcd_component)
        R22 = np.array([[np.cos(-angle), 0, np.sin(-angle)],
                    [0, 1, 0],
                    [-np.sin(-angle), 0, np.cos(-angle)]])       #y
        trans_pcd.rotate(R22,center = outer_box.get_center())
        return [component_angle, R2, axis]
    
    elif axis == 'z':
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])                                  #z
        trans_pcd.rotate(R3,center = outer_box.get_center()) 
        trans_pcd_component = calculate_pca(np.array(trans_pcd.points))
        target_pcd_component = calculate_pca(np.array(target_pcd.points))
        component_angle = calculate_angle(trans_pcd_component,target_pcd_component)
        R33 = np.array([[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]])                                  #z
        trans_pcd.rotate(R33,center = outer_box.get_center())
        return [component_angle, R3, axis]

### 5、基于点-面的精匹配
def read_file_original(pcd):
    data = np.array(pcd.points)
    return data

def read_file_deformed(pcd, normals):
    a = []
    for i,point in enumerate(pcd.points):
        x = point[0]
        y = point[1]
        z = point[2]       
        nx = normals[i][0]
        ny = normals[i][1]
        nz = normals[i][2]
        b = np.array([x,y,z,nx,ny,nz])
        a.append(b)                 
    data = np.array(a)
    return data                 

def icp_point_to_plane_lm(trans_points,trans_pcd,target_pcd,target_normals,initial,loop):
    T = np.identity(4)
    # 存储对应的最近邻点和法向量
    closest_points = []
    closest_normals = []
    for j,point in enumerate(trans_points):
        search_tree = o3d.geometry.KDTreeFlann(target_pcd)
        results = search_tree.search_knn_vector_3d(trans_pcd.points[j], 30)
        neighbor_indices = results[1]  # 最近邻索引列表 
        closest_points.append(target_pcd.points[neighbor_indices[0]])
        closest_normals.append(target_normals[neighbor_indices[0]])
    closest_points = np.array(closest_points)
    closest_normals = np.array(closest_normals)
    errors = closest_points - trans_points
    distances = np.einsum('ij,ij->i', errors, closest_normals)
    mean_distance = np.mean(distances)
    print("mean_distance:",abs(mean_distance))
    if abs(mean_distance) < 0.2:  
        return trans_pcd,T
    
    J = []
    e = []
    for i in range (0,np.array(target_pcd.points).shape[0]-1):
        dx = target_pcd.points[i][0]
        dy = target_pcd.points[i][1]
        dz = target_pcd.points[i][2]
        nx = target_normals[i][0]
        ny = target_normals[i][1]
        nz = target_normals[i][2]
        sx = trans_pcd.points[i][0]
        sy = trans_pcd.points[i][1]
        sz = trans_pcd.points[i][2]
        alpha = initial[0][0]
        beta = initial[1][0]
        gamma = initial[2][0]
        tx = initial[3][0]        
        ty = initial[4][0]
        tz = initial[5][0] 
        a1 = (nz*sy) - (ny*sz)
        a2 = (nx*sz) - (nz*sx)
        a3 = (ny*sx) - (nx*sy)
        a4 = nx
        a5 = ny
        a6 = nz
        _residual = (alpha*a1) + (beta*a2) + (gamma*a3) + (nx*tx) + (ny*ty) + (nz*tz) - (((nx*dx) + (ny*dy) + (nz*dz)) - ((nx*sx) + (ny*sy) + (nz*sz)))
        _J = np.array([a1, a2, a3, a4, a5, a6])
        _e = np.array([_residual])
    J.append(_J)
    e.append(_e)
        
    jacobian = np.array(J)
    residual = np.array(e)
    
    update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)
    initial = initial + update
    eular = []
    alpha = initial[0]
    belta = initial[1]
    gamma = initial[2]
    eular = [float(alpha),float(belta),float(gamma)]
    x = initial[3]
    y = initial[4]
    z = initial[5]
    t = np.transpose([float(x),float(y),float(z)])
    R = euler2rot(eular)
    # T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    trans_points = apply_transformation(trans_points,T)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans_points)
    # o3d.visualization.draw_geometries([trans_pcd, target_pcd])
    loop = loop + 1
    return icp_point_to_plane_lm(trans_points,trans_pcd,target_pcd,target_normals,initial,loop)

def calculate_normals(pcd):
    radius = 1  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.asarray(pcd.normals)
    # print("normals:",normals)
    # o3d.visualization.draw_geometries([pcd], window_name="法线估计",
    #                                 point_show_normal=True,
    #                                 width=800,  # 窗口宽度
    #                                 height=600)  # 窗口高度
    return normals

####欧拉角转旋转矩阵
def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

# 旋转矩阵转换为四元数
def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    print(f"x: {q.x}, y: {q.y}, z: {q.z}, w: {q.w}")
    return q.x,q.y,q.z,q.w

def main():
    '''
    参数说明：
        1、target_pcd:CREStereo重建出来的部分点云深度模型
        2、source_pcd:SOFTRAS重建出来的工件点云
    '''
    
    # 读取点云数据
    target_pcd = o3d.io.read_point_cloud('./output_model.pcd')
    # 设置下采样的体素尺寸
    voxel_size = 0.1
    # 进行下采样
    downsampled_points = downsample_point_cloud(target_pcd.points, voxel_size)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    # o3d.visualization.draw_geometries([target_pcd])

    points = target_pcd.points
    coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,
                                                                     origin=[0, 0, 0])  # Tag标签的模型 暂时以（0 0 0）为参考坐标系原点
    # o3d.visualization.draw_geometries([target_pcd, coord_frame2])  # 1

    world_points = camera_to_world(points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(world_points)

    source_pcd = o3d.io.read_point_cloud('./result.pcd')
    voxel_size = 0.1
    source_pcd_points = downsample_point_cloud(source_pcd.points, voxel_size)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)
    print("source_pcd点云数量:", len(source_pcd.points))
    # o3d.visualization.draw_geometries([source_pcd, target_pcd, coord_frame2])

    w1 = source_pcd.get_axis_aligned_bounding_box()
    w1.color = (1, 0, 0)
    extent1 = w1.get_extent()
    length1, width1, height1 = extent1[2], extent1[1], extent1[0]

    w2 = target_pcd.get_axis_aligned_bounding_box()
    w2.color = (0, 1, 0)
    extent2 = w2.get_extent()
    length2, width2, height2 = extent2[0], extent2[1], extent2[2]
    t = np.mean(target_pcd.points, axis=0) - np.mean(source_pcd.points, axis=0) + [0, (width1) / 2, 0]
    T = np.identity(4)
    T[:3, 3] = t
    trans_point = apply_transformation(np.array(source_pcd.points), T)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans_point)
    w3 = trans_pcd.get_axis_aligned_bounding_box()
    w3.color = (0, 1, 0)
    extent3 = w3.get_extent()
    length3, width3, height3 = extent3[2], extent3[1], extent3[0]

    T1 = T
    # o3d.visualization.draw_geometries([source_pcd, trans_pcd, target_pcd, coord_frame2, w1, w2, w3])

    ##PCA应用
    PI = math.pi
    information = []

    for i in range(36):
        jiaodu = (i + 1) * 10
        angle = (PI / 180) * jiaodu
        x_infor = pcd_revolve(target_pcd, trans_pcd, angle, 'x', w3)
        information.append(x_infor)
        y_infor = pcd_revolve(target_pcd, trans_pcd, angle, 'y', w3)
        information.append(y_infor)
        z_infor = pcd_revolve(target_pcd, trans_pcd, angle, 'z', w3)
        information.append(z_infor)

    min_angle = 1
    best_R = np.identity(3)
    best_axis = ''
    ###找到夹角最小的组合
    for part in information:
        if part[0] < min_angle:
            min_angle = part[0]
            best_R = part[1]
            best_axis = part[2]
    print('min_angle:', min_angle)
    print('best_R:', best_R)
    print('best_axis:', best_axis)
    best = pcd_revolve(target_pcd, trans_pcd, min_angle, best_axis, w3)
    trans_pcd.rotate(best_R, center=w3.get_center())
    ##记录T
    # T = np.identity(4)
    T1[:3, :3] = best_R
    # T2 = T
    # o3d.visualization.draw_geometries([trans_pcd, source_pcd, target_pcd, coord_frame2])

    target_normals = calculate_normals(target_pcd)
    trans_points = read_file_original(trans_pcd)
    dest_points_et_normal = read_file_deformed(target_pcd, target_normals)
    initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
    trans_pcd, T2 = icp_point_to_plane_lm(trans_points, trans_pcd, target_pcd, target_normals, initial, 0)

    ###6、输出信息
    fianl_T = np.dot(T2, T1)
    target_k = target_pcd.get_axis_aligned_bounding_box()
    target_k.color = (1, 0, 0)
    print("工件中心坐标为：", fianl_T[0:3, 3])
    trans_k = trans_pcd.get_axis_aligned_bounding_box()
    trans_k.color = (0, 1, 0)
    # o3d.visualization.draw_geometries([target_pcd, trans_pcd, coord_frame2])

    x, y, z, w = rotateToQuaternion(fianl_T[:3, :3])
    pingyi = fianl_T[:3, 3]
    print("siyuanshu:", x, y, z, w)
    print("pingyi:", pingyi)
    return x, y, z, w, pingyi

if __name__ == '__main__':
    x, y, z, w, pingyi = main()
