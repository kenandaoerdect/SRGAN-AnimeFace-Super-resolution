from sewar.full_ref import mse
from sewar.full_ref import psnr
import os
import cv2


def calculate_mse_psnr(fake_path, true_path):
    fake_imgs_name = os.listdir(fake_path)
    fake_imgs_name.sort()
    true_imgs_name = os.listdir(true_path)
    true_imgs_name.sort()
    assert len(fake_imgs_name) == len(true_imgs_name), '图片数量不匹配'
    MSE_list = []
    PSNR_list = []
    for idx in range(len(fake_imgs_name)):
        fake_arr = cv2.imread(os.path.join(fake_path, fake_imgs_name[idx]))
        true_arr = cv2.imread(os.path.join(true_path, true_imgs_name[idx]))
        MSE = mse(true_arr, fake_arr)
        PSNR = psnr(true_arr, fake_arr)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        print(fake_imgs_name[idx])
    return sum(MSE_list)/len(fake_imgs_name), sum(PSNR_list)/len(fake_imgs_name)


def main():
    fake_path = '../result/fake_hr_epoch_100'
    true_path = '../data/test/HR'
    avg_mse, avg_psnr = calculate_mse_psnr(fake_path, true_path)
    print('平均MSE：', avg_mse)
    print('平均PSNR：', avg_psnr )


if __name__ == '__main__':
    main()