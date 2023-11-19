from tools.config import *
import cv2
def load_images():
    print('receive photos')
    images = []
    for i in range(1,5):
        filename = f'imgs/{i}.jpg'
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
        else:
            print('无图片')
    return images

def s_rectification(images):
    print('开始立体校正')
    re_images = []
    for i in range(0,3):
        left = i+2
        right = i+1
        if i == 1:
            s = roi_config['group_{}_l'.format(i+1)][6]
            # s = 10
        else:
            s = roi_config['group_{}'.format(i+1)][6]

        left_camera_matrix = in_config['s{}_camera_matrix'.format(left)]
        left_distortion = in_config['s{}_distortion'.format(left)]

        right_camera_matrix = in_config['s{}_camera_matrix'.format(right)]
        right_distortion = in_config['s{}_distortion'.format(right)]

        img_left = images[left-1]
        img_right = images[right-1]
        h, w = img_left.shape[:2]

        #这一组要手动
        if i == 0:
            old_left_camera_matrix = left_camera_matrix
            left_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(old_left_camera_matrix, left_distortion, (w, h), 1,(w, h))
            img_left = cv2.undistort(img_left, old_left_camera_matrix, left_distortion, None, left_camera_matrix)

            old_right_camera_matrix = right_camera_matrix
            right_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(old_right_camera_matrix, right_distortion, (w, h), 1,(w, h))
            img_right = cv2.undistort(img_right, old_right_camera_matrix, right_distortion, None, right_camera_matrix)

            left_distortion = np.array([[0, 0, 0, 0, 0]])
            right_distortion = np.array([[ 0, 0, 0, 0, 0]])

        E = np.matmul(ex_config['s{}_T'.format(right)], np.linalg.inv(ex_config['s{}_T'.format(left)]))
        R = E[:3, :3]
        T = E[:3, 3]
        size = (2688 * s, 1520 * s)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                          right_camera_matrix, right_distortion, size,
                                                                          R,
                                                                          T, alpha=-1)
        print('Q:',Q)
        print('before:',T)
        print('after:',np.matmul(R2,T))
        print('R1:',R1,'\n')
        
        left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size,
                                                           cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                             cv2.CV_16SC2)

        left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
        right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

        # 截取感兴趣区域
        if i==1:
            ### 第二组图片中靠左边的工件
            roi_h_l = roi_config['group_2_l'][0]
            roi_w_l = roi_config['group_2_l'][1]
            roil_x_l = roi_config['group_2_l'][2]
            roil_y_l = roi_config['group_2_l'][3]
            roir_x_l = roi_config['group_2_l'][4]
            roir_y_l = roi_config['group_2_l'][5]
            roi_l_l = left[roil_y_l:roil_y_l+roi_h_l, roil_x_l:roil_x_l+roi_w_l]
            roi_r_l = right[roir_y_l:roir_y_l+roi_h_l, roir_x_l:roir_x_l+roi_w_l]
            re_images.append(roi_l_l)
            re_images.append(roi_r_l)
            cv2.imwrite('imgs/group_2_l_l.jpg',roi_l_l)
            cv2.imwrite('imgs/group_2_r_l.jpg',roi_r_l)

            roi_h_r = roi_config['group_2_r'][0]
            roi_w_r = roi_config['group_2_r'][1]
            roil_x_r = roi_config['group_2_r'][2]
            roil_y_r = roi_config['group_2_r'][3]
            roir_x_r = roi_config['group_2_r'][4]
            roir_y_r = roi_config['group_2_r'][5]
            roi_l_r = left[roil_y_r:roil_y_r+roi_h_r, roil_x_r:roil_x_r+roi_w_r]
            roi_r_r = right[roir_y_r:roir_y_r+roi_h_r, roir_x_r:roir_x_r+roi_w_r]
            re_images.append(roi_l_r)
            re_images.append(roi_r_r)
            cv2.imwrite('imgs/group_2_l_r.jpg',roi_l_r)
            cv2.imwrite('imgs/group_2_r_r.jpg',roi_r_r)
            
        else:
            roi_h = roi_config['group_{}'.format(i+1)][0]
            roi_w = roi_config['group_{}'.format(i+1)][1]
            roil_x =roi_config['group_{}'.format(i+1)][2]
            roil_y =roi_config['group_{}'.format(i+1)][3]
            roir_x =roi_config['group_{}'.format(i+1)][4]
            roir_y =roi_config['group_{}'.format(i+1)][5]
            roi_l = left[roil_y:roil_y + roi_h, roil_x:roil_x + roi_w]
            roi_r = right[roir_y:roir_y + roi_h, roir_x:roir_x + roi_w]
            re_images.append(roi_l)
            re_images.append(roi_r)
            cv2.imwrite('imgs/group{}_l.jpg'.format(i+1),roi_l)
            cv2.imwrite('imgs/group{}_r.jpg'.format(i+1),roi_r)
    return re_images

def main():
    loaded_images = load_images()
    if len(loaded_images) == 4:
        print('有4张图片')
        s_rectification_images = s_rectification(loaded_images)
    else:
        print('图片不够')

if __name__ == '__main__':
    main()
