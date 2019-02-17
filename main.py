import model
import numpy as np
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.utils import plot_model
from skimage import io, data, color


# gray_data_model_1.hdf5: 保持Transpose以及1 输入gray图  数据集是gray_data_1000.npy batch 30 查看画图保存
# gray_data_model_2.hdf5: 保持Transpose以及1 输入gray图  数据集是gray_data_1000.npy batch 14 查看画图保存
# data_mask_2000_model_1.hdf5: 保持Transpose以及1 输入gray图  数据集是data_mask_2000_.npy batch 50 查看画图保存
# data_mask_2000_model_2.hdf5: 保持Transpose以及1 输入gray图  数据集是data_mask_2000_.npy batch 10 查看画图保存 加入mIoU(失败了)
# data_mask_2000_model_3.hdf5: 保持Transpose以及1 输入gray图  数据集是data_mask_2000_.npy batch 50 查看画图保存 加入auto learning rate
# data_mask_2000_model_4.hdf5: 保持Transpose以及1 输入gray图  数据集是data_mask_2000_.npy batch 50 查看画图保存
# 加入auto learning rate 以及部分dropout(从上升开始)
# TO DO
# 优化predict.py (可以从数据读取到结果输出不需要手动每次调整)
# 每个文件有的可以写成变量模式 全部写成变量(不要手动输入)
# 可视化 每层activation 以及filter


class Main(object):

    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        train_image = np.load("data/data_2000_.npy")
        test_image = train_image
        train_image = train_image[0:1800].astype('float') / 255

        train_image_mask = np.load("data/data_mask_2000_.npy")
        test_image_mask = train_image_mask
        train_image_mask = train_image_mask[0:1800].astype('float') / 255

        test_image = test_image[1800:1810].astype('float') / 255

        test_image_mask = test_image_mask[1800:1810].astype('float') / 255

        return train_image, train_image_mask, test_image

    def train(self, output_name):
        print("-----------Loading data-------------")
        train_image, train_image_mask, test_image = self.load_data()
        print('-----------Loading data done--------')
        model1 = model.unet()
        # plot_model(model1, to_file='/home/xingyu/Desktop/model.png', show_shapes=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
        model_checkpoint = ModelCheckpoint("model/"+output_name+".hdf5", monitor='loss', verbose=1, save_best_only=True)
        print('-----------Fitting Model------------')
        history = model1.fit(
            train_image,
            train_image_mask,
            batch_size=4,
            epochs=50,
            verbose=1,
            validation_split=0.2,
            shuffle=True,
            callbacks=[model_checkpoint, reduce_lr],
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
        plt.savefig("plot/"+output_name+"-loss.png")

        plt.figure(num=2)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training & validation acc')
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig("plot/"+output_name+"-acc.png")

        print('-----------Predict test data--------')
        image_mask_test = model1.predict(test_image, batch_size=16, verbose=1)
        np.save("results/"+output_name+".npy", image_mask_test)

    # 需要写成自适应的阈值好一些
    def save_img(self, output_name):
        print('-----------Array to Image--------')
        imgs = np.load("results/"+output_name+".npy")
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = image.array_to_img(img)
            img.save("results/"+output_name+"-_%d.jpg" % (i))
            afterImg = io.imread("results/"+output_name+"-_%d.jpg" % (i))
            rows, cols = afterImg.shape
            out = 100
            for j in range(rows):
                for k in range(cols):
                    if afterImg[j, k] > out:
                        afterImg[j, k] = 255
                    else:
                        afterImg[j, k] = 0
            io.imsave("results/"+output_name+"-after-_%d.jpg" % (i), afterImg.astype(np.uint8))

if __name__ == '__main__':
    mynet = Main()
    output_name = 'data_mask_2000_model_4'
    mynet.train(output_name)
    mynet.save_img(output_name)
