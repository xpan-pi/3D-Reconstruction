###手动移动 粗匹配  TODO 后面计划将这个手动移动换成：先将两个点云的中心点移动到一起，然后将源点云每隔20度分别绕着x,y,z轴旋转，记录每次最近邻点小于阈值的点的个数，最后挑选出最佳的组合
# from sklearn.decomposition import PCA
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

def downsample_point_cloud(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd.points
def apply_transformation(source_points, transformation):
    source_points_homogeneous = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    transformed_points_homogeneous = np.dot(transformation, source_points_homogeneous.T).T
    transformed_points = np.divide(transformed_points_homogeneous[:, :3], transformed_points_homogeneous[:, 3].reshape(-1, 1))
    return transformed_points
# 旋转矩阵转换为四元数
def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    print(f"x: {q.x}, y: {q.y}, z: {q.z}, w: {q.w}")
    return q

# source_pcd = o3d.io.read_point_cloud('./result.ply')
# voxel_size = 0.1
# source_pcd_points = downsample_point_cloud(source_pcd.points, voxel_size)
# source_pcd = o3d.geometry.PointCloud()
# source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)
# print("source_pcd:",len(source_pcd.points))

# target_pcd = o3d.io.read_point_cloud('./67new_cloud.pcd')
# target_pcd_points = downsample_point_cloud(target_pcd.points, voxel_size)
# target_pcd = o3d.geometry.PointCloud()
# target_pcd.points = o3d.utility.Vector3dVector(target_pcd_points)
# print("target_pcd:",len(target_pcd.points))

# coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0]) #Tag标签的模型 暂时以（0 0 0）为参考坐标系原点
# o3d.visualization.draw_geometries([source_pcd, target_pcd,coord_frame2])

# angle = np.pi / 2  # 弧度制，相当于90度
# R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
#               [0, 1, 0],
#               [-np.sin(angle), 0, np.cos(angle)]]) #y

# w1 = source_pcd.get_axis_aligned_bounding_box()
# w1.color = (1,0,0)
# extent1=w1.get_extent()
# length1,width1,height1 = extent1[2],extent1[1],extent1[0] 

# t = np.mean(target_pcd.points, axis=0) - np.mean(source_pcd.points, axis=0)+[0,(width1)/2,0]
# T = np.identity(4)
# T[:3, :3] = R2
# T[:3, 3] = t
# trans_point = apply_transformation(np.array(source_pcd.points),T)

# source_pcd = o3d.geometry.PointCloud()
# source_pcd.points = o3d.utility.Vector3dVector(trans_point)

# T1 = T
# o3d.visualization.draw_geometries([source_pcd, target_pcd,coord_frame2])

########################基于点-面的精匹配
##点面匹配

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
        
def icp_point_to_plane_lm(source_points,source_pcd,target_pcd,target_normals,initial,loop):
    J = []
    e = []
    
    for i in range (0,np.array(target_pcd.points).shape[0]-1):
        dx = target_pcd.points[i][0]
        dy = target_pcd.points[i][1]
        dz = target_pcd.points[i][2]
        nx = target_normals[i][0]
        ny = target_normals[i][1]
        nz = target_normals[i][2]
        sx = source_pcd.points[i][0]
        sy = source_pcd.points[i][1]
        sz = source_pcd.points[i][2]
        
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
    
    # print("jacobian,residual:",jacobian,residual)
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
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    source_points = apply_transformation(source_points,T)
    after_pcd = o3d.geometry.PointCloud()
    after_pcd.points = o3d.utility.Vector3dVector(source_points)
    o3d.visualization.draw_geometries([after_pcd, target_pcd])
    
    loop = loop + 1
    # 存储对应的最近邻点和法向量
    closest_points = []
    closest_normals = []
    for j,point in enumerate(source_points):
        search_tree = o3d.geometry.KDTreeFlann(target_pcd)
        results = search_tree.search_knn_vector_3d(source_pcd.points[j], 30)
        neighbor_indices = results[1]  # 最近邻索引列表 
        closest_points.append(target_pcd.points[neighbor_indices[0]])
        closest_normals.append(target_normals[neighbor_indices[0]])
    closest_points = np.array(closest_points)
    closest_normals = np.array(closest_normals)
    errors = closest_points - source_points
    distances = np.einsum('ij,ij->i', errors, closest_normals)
    mean_distance = np.mean(distances)
    print("mean_distance:",abs(mean_distance))
    if abs(mean_distance) < 0.1:  
        return after_pcd,T
    # icp_point_to_plane_lm(source_points,source_pcd,target_pcd,target_normals,initial,loop)
    return icp_point_to_plane_lm(source_points,source_pcd,target_pcd,target_normals,initial,loop)

def downsample_point_cloud(point_cloud, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return downsampled_pcd.points

def calculate_normals(pcd):
    radius = 1  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.asarray(pcd.normals)
    # print("normals:",normals)
    o3d.visualization.draw_geometries([pcd], window_name="法线估计",
                                    point_show_normal=True,
                                    width=800,  # 窗口宽度
                                    height=600)  # 窗口高度
    return normals

####欧拉角转旋转矩阵
def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

# target_normals = calculate_normals(target_pcd)
# source_points = read_file_original(source_pcd)
# dest_points_et_normal = read_file_deformed(target_pcd,target_normals)
# initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
# trans_pcd,T2 = icp_point_to_plane_lm(source_points,source_pcd,target_pcd,target_normals ,initial,0)
# fianl_T = np.dot(T2,T1)
# target_k = target_pcd.get_axis_aligned_bounding_box()
# target_k.color = (1, 0, 0)
# print("工件坐标为：",fianl_T[0:3,3])
# trans_k = trans_pcd.get_axis_aligned_bounding_box()
# trans_k.color = (0, 1, 0)
# o3d.visualization.draw_geometries([target_pcd,trans_pcd,coord_frame2,target_k,trans_k])
# siyuanshu = rotateToQuaternion(fianl_T[:3, :3])
# print("siyuanshu:",siyuanshu)

def main():
    source_pcd = o3d.io.read_point_cloud('./result.ply')
    voxel_size = 0.1
    source_pcd_points = downsample_point_cloud(source_pcd.points, voxel_size)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pcd_points)
    print("source_pcd:",len(source_pcd.points))

    target_pcd = o3d.io.read_point_cloud('./67new_cloud.pcd')
    target_pcd_points = downsample_point_cloud(target_pcd.points, voxel_size)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pcd_points)
    print("target_pcd:",len(target_pcd.points))

    coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0]) #Tag标签的模型 暂时以（0 0 0）为参考坐标系原点
    o3d.visualization.draw_geometries([source_pcd, target_pcd,coord_frame2])

    angle = np.pi / 2  # 弧度制，相当于90度
    R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]]) #y

    w1 = source_pcd.get_axis_aligned_bounding_box()
    w1.color = (1,0,0)
    extent1=w1.get_extent()
    length1,width1,height1 = extent1[2],extent1[1],extent1[0] 

    t = np.mean(target_pcd.points, axis=0) - np.mean(source_pcd.points, axis=0)+[0,(width1)/2,0]
    T = np.identity(4)
    T[:3, :3] = R2
    T[:3, 3] = t
    trans_point = apply_transformation(np.array(source_pcd.points),T)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(trans_point)

    T1 = T
    o3d.visualization.draw_geometries([source_pcd, target_pcd,coord_frame2])

    target_normals = calculate_normals(target_pcd)
    source_points = read_file_original(source_pcd)
    dest_points_et_normal = read_file_deformed(target_pcd,target_normals)
    initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
    trans_pcd,T2 = icp_point_to_plane_lm(source_points,source_pcd,target_pcd,target_normals ,initial,0)
    fianl_T = np.dot(T2,T1)
    target_k = target_pcd.get_axis_aligned_bounding_box()
    target_k.color = (1, 0, 0)
    print("工件坐标为：",fianl_T[0:3,3])
    trans_k = trans_pcd.get_axis_aligned_bounding_box()
    trans_k.color = (0, 1, 0)
    o3d.visualization.draw_geometries([target_pcd,trans_pcd,coord_frame2,target_k,trans_k])
    print(fianl_T[:3, :3])
    siyuanshu = rotateToQuaternion(fianl_T[:3, :3])
    print("siyuanshu:",siyuanshu)

if __name__ == '__main__':
    main()
