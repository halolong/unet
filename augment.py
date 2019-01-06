import unet

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
'''
# datagen & rescale
image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

base_dir = '/home/xingyu/Desktop/Building Segmentation using U-net/data/train'
image_dir = os.path.join(base_dir, 'image')
mask_dir = os.path.join(base_dir, 'mask')
# Test NOT READY

# define generator
image_generator = image_datagen.flow_from_directory(
    'data/image',
    target_size=(256, 256),
    batch_size=10,
    class_mode='binary')

mask_generator = mask_datagen.flow_from_directory(
    'data/mask',
    target_size=(256, 256),
    batch_size=10,
    class_mode='binary')


# create a new generator that includes mask and image generator
train_generator = zip(image_generator, mask_generator)

myNet = Building Segmentation using U-net.myUnet()
model = myNet.get_unet()
model_checkpoint = ModelCheckpoint('model/data-flow.hdf5', monitor='loss', verbose=1, save_best_only=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=10,
    callbacks=[model_checkpoint]
)

data = np.load('data/data_2000_.npy')
test = data[0:10]
res = model.predict(test, batch_size=5, verbose=1)
k = 0
for i in res:
    img = image.array_to_img(i)
    img.save('data_flow_%d.tif' % k)
    k = k + 1

报错显示为除数为0.... 目前不知道为啥
'''


img = np.load("data/data_1000_.npy")
img = img.astype(float) / 255

mask = np.load('data/data_mask_1000_.npy')
mask = mask.astype(float) / 255

datagen = ImageDataGenerator(
    width_shift_range=0.2,
    rotation_range=20,
)

datagen.fit(img)
datagen.fit(mask)

myNet = unet.myUnet()
model = myNet.get_unet()
# model_checkpoint = ModelCheckpoint('model/flow.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(datagen.flow(img, mask, batch_size=4),
                              steps_per_epoch=1000/4,
                              epochs=100)
print(history.keys())

