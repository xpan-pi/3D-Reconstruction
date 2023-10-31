from tools.config import *
import cv2
def load_images():
    print('receive photos')
    images = []
    for i in range(1,9):
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
    for img in images:
        print(img.shape)

def main():
    loaded_images = load_images()
    if len(loaded_images) == 8:
        print('有8张图片')
        s_rectification(loaded_images)
    else:
        print('图片不够')

if __name__ == '__main__':
    main()
