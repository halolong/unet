import model
import numpy as np
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import plot_model


# Building Segmentation using U-net-1: 修改了upsampling以及3
# Building Segmentation using U-net-2: 保持Transpose以及3
# Building Segmentation using U-net-3: 保持umsampling以及1 输入gray图
# Building Segmentation using U-net-4: 保持Transpose以及1 输入gray图
# Building Segmentation using U-net-5: 保持Transpose以及1 输入gray图  数据集是data_2000_ (这里的2000是指1000-2000的数据集)
# Building Segmentation using U-net-6: 保持Transpose以及1 输入gray图  数据集是data_1000_ 加大batch 30 查看画图保存
# Building Segmentation using U-net-7: 保持Transpose以及1 输入gray图  数据集是gray_data_1000_ 加大batch 30 查看画图保存
# Building Segmentation using U-net-8: 保持Transpose以及1 输入gray图  数据集是data_2000_ 加大batch 30 查看画图保存

class Main(object):
    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        train_image = np.load("data/data_zjy_.npy")
        test_image = train_image
        train_image = train_image[0:60].astype('float') / 255

        train_image_mask = np.load("data/data_mask_zjy_.npy")
        test_image_mask = train_image_mask
        train_image_mask = train_image_mask[0:60].astype('float') / 255

        test_image = test_image[55:60].astype('float') / 255

        test_image_mask = test_image_mask[55:60].astype('float') / 255

        return train_image, train_image_mask, test_image

    def train(self):
        print("-----------Loading data-------------")
        train_image, train_image_mask, test_image = self.load_data()
        print('-----------Loading data done--------')
        model1 = model.unet()
        # plot_model(model1, to_file='/home/xingyu/Desktop/model.png', show_shapes=True)
        model_checkpoint = ModelCheckpoint('model/Building Segmentation using U-net-zjy-2.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('-----------Fitting Model------------')
        history = model1.fit(
            train_image,
            train_image_mask,
            batch_size=8,
            epochs=100,
            verbose=1,
            validation_split=0.2,
            shuffle=True,
            callbacks=[model_checkpoint]
        )
        print('-----------Plotting-----------------')
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.figure(num=1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training & validation loss')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plot/model-zjy-2-loss.png')

        plt.figure(num=2)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training & validation acc')
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig('plot/model-zjy-2-acc.png')

        print('-----------Predict test data--------')
        image_mask_test = model1.predict(test_image, batch_size=16, verbose=1)
        np.save('results/images_mask_test_zjy-2.npy', image_mask_test)

    def save_img(self):
        print('-----------Array to Image--------')
        imgs = np.load('results/images_mask_test_zjy-2.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = image.array_to_img(img)
            img.save("results/Building Segmentation using U-net-zjy-2-_%d.jpg" % (i))


if __name__ == '__main__':
    mynet = Main()
    mynet.train()
    mynet.save_img()