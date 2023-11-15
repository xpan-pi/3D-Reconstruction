import os
import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2
import sys
sys.path.append('/home/ecust/sanweichonjian/CREStereo-master/CREStereo-master')
from nets import Model
import open3d as o3d
from tools.config import *

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
#获取模型
model_path = '/home/ecust/sanweichonjian/3D-Reconstruction-main/crestereo_eth3d.mge'
model_func = load_model(model_path)

for i in range(1,4):
    if i == 2:
        left_r = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group_2_l_r.jpg')
        right_r = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group_2_r_r.jpg')
        
        left_l = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group_2_l_l.jpg')
        right_l = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group_2_r_l.jpg')
        in_h, in_w = left_r.shape[:2]
        ####获取视差图 右边那个工件
        size = size_config['group2_r_size']   
        eval_h, eval_w = [int(e) for e in size.split("x")]
        left_img_r = cv2.resize(left_r, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img_r = cv2.resize(right_r, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        pred = inference(left_img_r, right_img_r, model_func, n_iter=10) #得到视差图
        ###还原视差图 将视差图像素变成和之前一样的
        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR)*t
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis_img = disp_vis.astype("uint8")
        disp_vis_img = cv2.resize(disp_vis_img,(546,196))

        cv2.imshow('disp_vis_img',disp_vis_img)
        cv2.imwrite('depth23_l.png',disp_vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        ####获取视差图 左边那个工件
        size = size_config['group2_l_size']  
        eval_h, eval_w = [int(e) for e in size.split("x")]
        left_img_l = cv2.resize(left_l, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img_l = cv2.resize(right_l, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        pred = inference(left_img_l, right_img_l, model_func, n_iter=10) #得到视差图
        ###还原视差图 将视差图像素变成和之前一样的
        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR)*t
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis_img = disp_vis.astype("uint8")
        disp_vis_img = cv2.resize(disp_vis_img,(546,196))

        cv2.imshow('disp_vis_img',disp_vis_img)
        cv2.imwrite('depth23_r.png',disp_vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        left = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group{}_l.jpg'.format(i))
        right = cv2.imread('/home/ecust/sanweichonjian/3D-Reconstruction-main/imgs/group{}_r.jpg'.format(i))
        in_h, in_w = left.shape[:2]
        ####获取视差图
        size = size_config['group{}_size'.format(i)]
        eval_h, eval_w = [int(e) for e in size.split("x")]
        left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        pred = inference(left_img, right_img, model_func, n_iter=10) #得到视差图
        ###还原视差图 将视差图像素变成和之前一样的
        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR)*t
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis_img = disp_vis.astype("uint8")
        disp_vis_img = cv2.resize(disp_vis_img,(546,196))
        cv2.imshow('disp_vis_img',disp_vis_img)
        cv2.imwrite('depth{}{}.png'.format(i,i+1),disp_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()