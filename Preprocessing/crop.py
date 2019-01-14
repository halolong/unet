# coding=utf-8
import cv2
import numpy as np
from PIL import Image
import os


# return file path of dir and file name list
def getfile(file_dir):
    path = ''
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # get the `postfix`
            # print(os.path.splitext(file)[1])
            filename = os.path.splitext(file)
            # print(filename)
            file_list.append(filename[0] + filename[1])
            # !!! need to change the .xxx
            if os.path.splitext(file)[1] == '.tif':
                path = os.path.join(root)
    # print(file_list)
    return path, file_list


# Parameter:file_path, flag(origin or mask), size, col_step, row_step
# Return: sum of pics
def crop(file_path, crop_size, col_step, row_step, out_path, identity, flag=''):
    # 读取图像
    img = Image.open(file_path)
    size = img.size
    pic_sum = 0
    row = 0
    col = 0

    for i in range(0, size[1] - crop_size, row_step):
        row = row + 1
        for j in range(0, size[0] - crop_size, col_step):
            col = col + 1
            # 横纵坐标需要好好分清
            region = img.crop((j, i, j + crop_size, i + crop_size))
            region.save(out_path + '%d.tif' % pic_sum)
            # region.save(out_path + '%d_%d_%s.tif' % (identity, pic_sum, flag))

            print('图片%d_%d.jpeg剪裁成功' % (row, col))
            pic_sum = pic_sum + 1
        col = 0
    print('一共%d张图片' % pic_sum)


def get_sort(string):
    return sorted(string, key=str.lower)


# method: root + filename
info = getfile('/home/xingyu/Desktop/pics/tif')
root = info[0]

# mask_root = '/home/xingyu/Desktop/InriaDataset/AerialImageDataset/train/gt'
name = info[1]
# print(name)
sort_name = get_sort(name)

for i in range(len(sort_name)):
    print(sort_name[i])
    crop(root+'/'+sort_name[i], 256, 150, 150, '/home/xingyu/Desktop/pics/test/', i)
    # crop(mask_root+'/'+sort_name[i], 256, 150, 150, '/home/xingyu/Desktop/result/', i, 'mask')
