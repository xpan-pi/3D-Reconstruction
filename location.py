import os
import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2
import sys
sys.path.append('/home/xpan/Desktop/program/CREStereo-master')
from nets import Model
import open3d as o3d
from tools.config import *
model_path = '/home/xpan/Desktop/program/CREStereo-master/crestereo_eth3d.mge'

###函数定义  加载模型
def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model

### 函数定义   将左右图像输入模型中进行前向传播 最终返回预测的视差图
def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    # print(left.shape)
    imgL = left.transpose(2, 0, 1)
    # print(imgL.shape)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])
    # 将图像转化为MegEngine张量 适配深度学习所需要的格式和数据类型
    imgL = mge.tensor(imgL).astype("float32")
    imgR = mge.tensor(imgR).astype("float32")
    #双线插值 将图像缩小一半
    imgL_dw2 = F.nn.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.nn.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy() #将最后结果转化为视差图

    return pred_disp

def output_model(group):
    print('hello')
    #获取模型
    model_func = load_model(model_path)
    ###模型推理
    left = cv2.imread('./imgs/group{}_l.jpg'.format(group))
    right = cv2.imread('./imgs/group{}_r.jpg'.format(group))
    in_h, in_w = left.shape[:2]
    print("in_h:",in_h)
    print("in_w:",in_w)
    ####获取视差图
    # size = '200x680' # 800x1360  400x680 200x680
    size = '800x1360' # 800x1360  400x680 200x680
    # size = size_config['group1_size']

    eval_h, eval_w = [int(e) for e in size.split("x")]
    left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    pred = inference(left_img, right_img, model_func, n_iter=10) #得到视差图

    ###还原视差图 将视差图像素变成和之前一样的
    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR)*t

    disp_h,disp_w = disp.shape[:2]
    print("disp_h:",disp_h)
    print("disp_w:",disp_w)

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis_img = disp_vis.astype("uint8")

    disp_vis_img = cv2.resize(disp_vis_img,(546,196))
    cv2.imshow('disp_vis_img',disp_vis_img)
    # cv2.imwrite('./imgs/depth12.png',disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # group_6 = [2353.19895,11888.8298,3969.47872,-8.953820] 

    ##### 得到工件稠密点云模型
    ######################
    #reconstruction
    #读取语义分割图像
    # img = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group5_l_labelme.png')
    img = cv2.imread('./imgs/group6.png')
    f = new_in_config ['group_{}'.format(group)][0]
    cx = new_in_config ['group_{}'.format(group)][1]
    cy = new_in_config ['group_{}'.format(group)][2]
    b = new_in_config ['group_{}'.format(group)][3]

    dis_x = roi_config['group_{}'.format(group)][2]
    dis_y = roi_config['group_{}'.format(group)][3]
    todo = roi_config['group_{}'.format(group)][2] - roi_config['group_{}'.format(group)][4]

    point_cloud = o3d.geometry.PointCloud()
    points = []
    depth_max = []
    for i in range(in_w):
        for j in range(in_h):
            if img[j][i][2] != 0:
                disp[j][i] = disp[j][i] + todo #TODO 2750是视差弥补 截取的左图的左上角-右图左上角
                depth = (b*f)/(-disp[j][i])  #要大一些
                x = (i+dis_x-cx)*b/(-disp[j][i])
                y = (j+dis_y-cy)*b/(-disp[j][i])
                n = np.array([x,y,depth])
                points.append(n)
                # disp[j][i] = 0
    np.array(points)
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

    # num_neighbors = 10 # K邻域点的个数
    # std_ratio = 1.0 # 标准差乘数
    # sor_pcd23, ind = point_cloud.remove_statistical_outlier(num_neighbors, std_ratio)
    wk1 = point_cloud.get_axis_aligned_bounding_box()
    wk1.color = (1, 0, 0)
    print("工件坐标：")
    print((wk1.get_max_bound()-wk1.get_min_bound()))
    o3d.visualization.draw_geometries([point_cloud,wk1])
    o3d.io.write_point_cloud('./output_model.pcd', point_cloud)


if __name__ == '__main__':
    print('hello')
    group = 6
    output_model(group)
