import cv2
import os

HR_img_path = '../data/test/HR'
LR_img_path = '../data/test/LR'

for img in os.listdir(HR_img_path):
    img_arr = cv2.imread(os.path.join(HR_img_path, img))
    resize_img = cv2.resize(img_arr, (img_arr.shape[0]//2, img_arr.shape[0]//2))
    cv2.imwrite(os.path.join(LR_img_path, img), resize_img)
    print(img)