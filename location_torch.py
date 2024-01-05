import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
sys.path.append('/home/ecust/Downloads/CREStereo-Pytorch-main')
from net import Model
device = 'cuda'
import open3d as o3d
from tools.config import *

def inference(left, right, model, n_iter=20):
	print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
	return pred_disp

def output_model(group):
    # model_path = '/home/ecust/Downloads/CREStereo-Pytorch-main/models/crestereo_eth3d.pth'
    # #获取模型
    # model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    # model.load_state_dict(torch.load(model_path), strict=True)
    # model.to(device)
    # model.eval()
    
    # ###模型推理
    # left = cv2.imread('./imgs/group{}_l.jpg'.format(group))
    # right = cv2.imread('./imgs/group{}_r.jpg'.format(group))
    # in_h, in_w = left.shape[:2]
    # # size = '200x680' # 800x1360  400x680 200x680
    # size = '800x1360' # 800x1360  400x680 200x680
    # eval_h, eval_w = [int(e) for e in size.split("x")]
    # left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # pred = inference(left_img, right_img, model, n_iter=10) #得到视差图
    # t = float(in_w) / float(eval_w)
    # disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR)*t

    # disp_h,disp_w = disp.shape[:2]
    # disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    # disp_vis_img = disp_vis.astype("uint8")
    # disp_vis_img = cv2.resize(disp_vis_img,(546,196))
    
    
    
    left_img = cv2.imread('./imgs/group{}_l.jpg'.format(group))
    right_img = cv2.imread('./imgs/group{}_r.jpg'.format(group))
    in_h, in_w = left_img.shape[:2]

    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h,in_w)
    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    model_path = '/home/ecust/Downloads/CREStereo-Pytorch-main/models/crestereo_eth3d.pth'
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()
    pred = inference(imgL, imgR, model, n_iter=20)
    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    # disp_vis = disp_vis.astype("uint8")
    # disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    ##### 得到工件稠密点云模型
    ######################
    img = cv2.imread('./semantic_imgs_out/group6_l.jpg')
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
    # o3d.visualization.draw_geometries([point_cloud])

    # num_neighbors = 10 # K邻域点的个数
    # std_ratio = 1.0 # 标准差乘数
    # sor_pcd23, ind = point_cloud.remove_statistical_outlier(num_neighbors, std_ratio)
    wk1 = point_cloud.get_axis_aligned_bounding_box()
    wk1.color = (1, 0, 0)
    # print("工件坐标：")
    # print((wk1.get_max_bound()-wk1.get_min_bound()))
    # o3d.visualization.draw_geometries([point_cloud,wk1])
    o3d.io.write_point_cloud('./output_model.pcd', point_cloud)
    print('location is done!')

if __name__ == '__main__':
    print('hello')
    group = 6
    output_model(group)
