import os, shutil

# original dataset path
original_dataset_dir = '/home/xingyu/Desktop/result'

# create small dataset
base_dir = '/home/xingyu/Desktop/small_result'
# os.mkdir(base_dir)

# Directories for training, validation, test
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

# x_x_(mask).tif
# 0_1023_ - 10_1023 train 10000 pics
# 10_0_ - 10_1023 test 1000 pics
origins = []
masks = []
# for i in range(10):
#     for j in range(1024):
#         origin = '%d_%d_.tif' % (i, j)
#         origins.append(origin)
#         mask = '%d_%d_mask.tif' % (i, j)
#         masks.append(mask)
#
# for element in origins:
#     src = os.path.join(original_dataset_dir, element)
#     dst = os.path.join(train_dir+'/image', element)
#     shutil.copyfile(src, dst)
#
# for element in masks:
#     src = os.path.join(original_dataset_dir, element)
#     dst = os.path.join(train_dir+'/mask', element)
#     shutil.copyfile(src, dst)
# print(len(masks))

# for test
for j in range(20):
    origin = '%d_%d_.tif' % (20, j)
    origins.append(origin)
    mask = '%d_%d_mask.tif' % (20, j)
    masks.append(mask)

for element in origins:
    src = os.path.join(original_dataset_dir, element)
    dst = '/home/xingyu/Desktop/small_result/validation/image/'+element
    print(src, dst)
    shutil.copyfile(src, dst)

for element in masks:
    src = os.path.join(original_dataset_dir, element)
    dst = '/home/xingyu/Desktop/small_result/validation/mask/'+element
    shutil.copyfile(src, dst)
