from tools.config import *
import cv2
def load_images():
    print('receive photos')
    images = []
    for i in range(6,8):
        filename = f'imgs/{i}.jpg'
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
        else:
            print('无图片') 
    return images

# 鼠标事件回调函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def stereo_rect(group):
    left = group
    right = group + 1
    s = roi_config['group_{}'.format(group)][6]
    left_camera_matrix = in_config['s{}_camera_matrix'.format(left)]
    left_distortion = in_config['s{}_distortion'.format(left)]
    right_camera_matrix = in_config['s{}_camera_matrix'.format(right)]
    right_distortion = in_config['s{}_distortion'.format(right)]

    loaded_images = load_images()
    img_left = loaded_images[0]
    img_right = loaded_images[1]
    h, w = img_left.shape[:2]

    E = np.matmul(ex_config['s{}_T'.format(right)], np.linalg.inv(ex_config['s{}_T'.format(left)]))
    R = E[:3, :3]
    T = E[:3, 3]
    size = (2688 * s, 1520 * s)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size,
                                                                  R,
                                                                  T, alpha=-1)
    # print(Q)
    # print('after:',np.matmul(R2,T))
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size,
                                                   cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                     cv2.CV_16SC2)
    left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
    right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)
    roi_h = roi_config['group_{}'.format(6)][0]  
    roi_w = roi_config['group_{}'.format(6)][1]  
    roil_x =roi_config['group_{}'.format(6)][2]
    roil_y =roi_config['group_{}'.format(6)][3]
    roir_x =roi_config['group_{}'.format(6)][4]
    roir_y =roi_config['group_{}'.format(6)][5]
    roi_l = left[roil_y:roil_y + roi_h, roil_x:roil_x + roi_w]
    roi_r = right[roir_y:roir_y + roi_h, roir_x:roir_x + roi_w]

    test_h,test_w = roi_l.shape[:2]
    print("roi_l_h:",test_h)
    print("roi_l_w:",test_w)
    img1 = cv2.resize(roi_l,(256,256))
    img2 = cv2.resize(roi_r,(256,256))
    cv2.imshow('left',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('right',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('./imgs/group{}_l.jpg'.format(group),roi_l)
    cv2.imwrite('./imgs/group{}_r.jpg'.format(group),roi_r)
    return 0

if __name__ == '__main__':
    print('hello')
    group = 6
    stereo_rect(group)
