from skimage import io, data, color
import numpy as np

img = io.imread('/home/xingyu/Desktop/Building Segmentation using U-net/results/Building Segmentation using U-net-5-_9.jpg')
#
rows, cols = img.shape
io.imsave('test_before.tif', img)
out = 100

for i in range(rows):
    for j in range(cols):
        if img[i, j] > out:
            img[i, j] = 255
        else:
            img[i, j] = 0
# for i in range(rows):
#     for j in range(cols):
#         print(img[i, j])
# img = img * 255
io.imsave('test_after.tif', img.astype(np.uint8))

