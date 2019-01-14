import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


# Converting pics (gray) to Numpy array
# sum: 10000 ( 10 npy file)
height = 256
width = 256
size = 15


def create_data():
    print('-' * 30)
    print('Creating data images...')
    print('-' * 30)

    base_dir = '/home/xingyu/Desktop/pics/test'

    data_origin_dir = os.path.join(base_dir)
    # data_mask_dir = os.path.join(base_dir, 'label')

    data_set = [os.path.join(data_origin_dir, fname) for fname in os.listdir(data_origin_dir)]
    sort_data_set = sorted(data_set)

    # data_mask_set = [os.path.join(data_mask_dir, fname) for fname in os.listdir(data_mask_dir)]
    # sort_mask_set = sorted(data_mask_set)

    imgdata = np.ndarray((size, height, width, 1), dtype=np.uint8)
    # imgmask = np.ndarray((size, height, width, 1), dtype=np.uint8)

    # for i in range(size):
    #     print(sort_mask_set[i])

    for i in range(size):
        img = image.load_img(sort_data_set[i], target_size=(height, width), grayscale=True)
        # mask = image.load_img(sort_mask_set[i], target_size=(height, width), grayscale=True)

        img = image.img_to_array(img)
        # mask = image.img_to_array(mask)

        imgdata[i] = img
        # imgmask[i] = mask

    np.save('/home/xingyu/Desktop/pics/test/data_test_.npy', imgdata)
    # np.save('../data/data_mask_zjy_.npy', imgmask)


create_data()

'''
def create_data():
    print('-' * 30)
    print('Creating data images...')
    print('-' * 30)

    base_dir = '/home/xingyu/Desktop/small_result'

    data_origin_dir = os.path.join(base_dir, 'train/image')
    data_mask_dir = os.path.join(base_dir, 'train/mask')

    data_set = [os.path.join(data_origin_dir, fname) for fname in os.listdir(data_origin_dir)]
    sort_data_set = sorted(data_set)

    data_mask_set = [os.path.join(data_mask_dir, fname) for fname in os.listdir(data_mask_dir)]
    sort_mask_set = sorted(data_mask_set)

    imgdata = np.ndarray((size, height, width, 1), dtype=np.uint8)
    imgmask = np.ndarray((size, height, width, 1), dtype=np.uint8)

    for i in range(size):
        print(sort_mask_set[i])

    for i in range(size):
        img = image.load_img(sort_data_set[i], target_size=(height, width), grayscale=True)
        mask = image.load_img(sort_mask_set[i], target_size=(height, width), grayscale=True)

        img = image.img_to_array(img)
        mask = image.img_to_array(mask)

        imgdata[i] = img
        imgmask[i] = mask

    np.save('../data/data_3000_.npy', imgdata)
    np.save('../data/data_mask_3000_.npy', imgmask)


create_data()

'''