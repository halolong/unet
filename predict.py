from keras import models
from keras.preprocessing import image

import numpy as np

test_data = np.load('data/data_3000_.npy')
test_data = test_data[2000:3000]
test_image = image.array_to_img(test_data[150])
test_image.save('/home/xingyu/Desktop/test.tif')
model = models.load_model('model/unet7.hdf5')
results = model.predict(test_data,
                       batch_size=2)

img = image.array_to_img(results[150])
img.save('/home/xingyu/Desktop/mask.tif')
# np.save('/home/xingyu/Desktop/pics/test/results.npy', results)

# print('---------array to image -------------')
# imgs = np.load('/home/xingyu/Desktop/pics/test/results.npy')
# for i in range(imgs.shape[0]):
#     img = imgs[i]
#     img = image.array_to_img(img)
#     img.save('/home/xingyu/Desktop/pics/test/result_%d.tif' % i)

