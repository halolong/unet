from keras import models, layers
from keras.utils import plot_model
from MeanIoU import MeanIoU
import numpy as np
import keras.backend as K


def iou(y_true, y_pred, label:int):
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection/union)


def mean_iou2(y_true, y_pred):
    num_labels = K.int_shape(y_pred)[-1] - 1
    mean_iou = K.variable(0)
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)
    return mean_iou / num_labels


def unet(input_size=(256, 256, 1)):
    input1 = layers.Input(input_size)

    # (1) CONV, RELU
    conv1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(input1)
    # print('conv1 output shape: ', conv1.shape)
    # (2) BN_CONV_RELU * 2, MaxPooling
    batch1 = layers.BatchNormalization(momentum=0.01)(conv1)  # affine
    conv2 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch1)
    copy1 = conv2
    batch1 = layers.BatchNormalization(momentum=0.01)(conv2)  # affine
    conv2 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch1)
    maxpool1 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(conv2)
    print('maxpooling1 output shape: ', maxpool1.shape)

    # (3) BN_CONV_RELU * 3, MaxPooling
    batch2 = layers.BatchNormalization(momentum=0.01)(maxpool1)  # affine
    conv3 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch2)
    batch2 = layers.BatchNormalization(momentum=0.01)(conv3)  # affine
    conv3 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch2)
    copy2 = conv3
    batch2 = layers.BatchNormalization(momentum=0.01)(conv3)  # affine
    conv3 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch2)
    maxpool2 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(conv3)
    print('maxpooling2 output shape: ', maxpool2.shape)

    # (4) BN_CONV_RELU * 3, MaxPooling
    batch3 = layers.BatchNormalization(momentum=0.01)(maxpool2)  # affine
    conv4 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch3)
    batch3 = layers.BatchNormalization(momentum=0.01)(conv4)  # affine
    conv4 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch3)
    copy3 = conv4
    batch3 = layers.BatchNormalization(momentum=0.01)(conv4)  # affine
    conv4 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch3)
    maxpool3 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(conv4)
    print('maxpooling3 output shape: ', maxpool3.shape)

    # (5) BN_CONV_RELU * 3, MaxPooling
    batch4 = layers.BatchNormalization(momentum=0.01)(maxpool3)  # affine
    conv5 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch4)
    batch4 = layers.BatchNormalization(momentum=0.01)(conv5)  # affine
    conv5 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch4)
    copy4 = conv5
    batch4 = layers.BatchNormalization(momentum=0.01)(conv5)  # affine
    conv5 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch4)
    maxpool4 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(conv5)
    print('maxpooling4 output shape: ', maxpool4.shape)

    # (6) BN_CONV_RELU * 3, MaxPooling
    batch5 = layers.BatchNormalization(momentum=0.01)(maxpool4)  # affine
    conv6 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch5)
    batch5 = layers.BatchNormalization(momentum=0.01)(conv6)  # affine
    conv6 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch5)
    copy5 = conv6
    batch5 = layers.BatchNormalization(momentum=0.01)(conv6)  # affine
    conv6 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch5)
    maxpool5 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(conv6)
    print('maxpooling5 output shape: ', maxpool5.shape)

    # (7) BN_CONV_RELU * 2, BN_UPCONV_RELU
    batch6 = layers.BatchNormalization(momentum=0.01)(maxpool5)  # affine
    conv7 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch6)
    batch6 = layers.BatchNormalization(momentum=0.01)(conv7)  # affine
    conv7 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch6)
    batch6 = layers.BatchNormalization(momentum=0.01)(conv7)  # affine
    upconv1 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(batch6)
    # upconv1 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(
    #     layers.UpSampling2D(size=(4, 4))(batch6))
    print('upCon1 output shape: ', upconv1.shape)

    # (8) CONCAT, BN_CONV_RELU * 2, BN_UPCONV_RELU
    concat1 = layers.concatenate([upconv1, copy5])
    batch7 = layers.BatchNormalization(momentum=0.01)(concat1)  # affine
    conv8 = layers.Conv2D(96, 3, strides=(1, 1), padding='same', activation='relu')(batch7)
    batch7 = layers.BatchNormalization(momentum=0.01)(conv8)  # affine
    conv8 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch7)
    batch7 = layers.BatchNormalization(momentum=0.01)(conv8)  # affine
    upconv2 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(batch7)
    # upconv2 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(
    #     layers.UpSampling2D(size=(4, 4))(batch7))
    print('upCon2 output shape: ', upconv2.shape)

    # (9) CONCAT, BN_CONV_RELU * 2, BN_UPCONV_RELU
    concat2 = layers.concatenate([upconv2, copy4])
    batch8 = layers.BatchNormalization(momentum=0.01)(concat2)  # affine
    conv9 = layers.Conv2D(96, 3, strides=(1, 1), padding='same', activation='relu')(batch8)
    batch8 = layers.BatchNormalization(momentum=0.01)(conv9)  # affine
    conv9 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch8)
    batch8 = layers.BatchNormalization(momentum=0.01)(conv9)  # affine
    upconv3 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(batch8)
    # upconv3 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(
    #     layers.UpSampling2D(size=(4, 4))(batch8))
    print('upCon3 output shape: ', upconv3.shape)

    # (10) CONCAT, BN_CONV_RELU * 2, BN_UPCONV_RELU
    concat3 = layers.concatenate([upconv3, copy3])
    batch9 = layers.BatchNormalization(momentum=0.01)(concat3)  # affine
    conv10 = layers.Conv2D(96, 3, strides=(1, 1), padding='same', activation='relu')(batch9)
    batch9 = layers.BatchNormalization(momentum=0.01)(conv10)  # affine
    conv10 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch9)
    batch9 = layers.BatchNormalization(momentum=0.01)(conv10)  # affine
    upconv4 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(batch9)
    # upconv4 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(
    #         layers.UpSampling2D(size=(4, 4))(batch9))
    print('upCon4 output shape: ', upconv4.shape)

    # (11) CONCAT, BN_CONV_RELU * 2, BN_UPCONV_RELU
    concat4 = layers.concatenate([upconv4, copy2])
    batch10 = layers.BatchNormalization(momentum=0.01)(concat4)  # affine
    conv11 = layers.Conv2D(96, 3, strides=(1, 1), padding='same', activation='relu')(batch10)
    batch10 = layers.BatchNormalization(momentum=0.01)(conv11)  # affine
    conv11 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch10)
    batch10 = layers.BatchNormalization(momentum=0.01)(conv11)  # affine
    upconv5 = layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(batch10)
    # upconv5 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(
    #         layers.UpSampling2D(size=(4, 4))(batch10))
    print('upCon5 output shape: ', upconv5.shape)

    # (12) CONCAT, BN_CONV_RELU * 2, CONVOUT, SIGMOID
    concat5 = layers.concatenate([upconv5, copy1])
    batch11 = layers.BatchNormalization(momentum=0.01)(concat5)  # affine
    conv12 = layers.Conv2D(96, 3, strides=(1, 1), padding='same', activation='relu')(batch11)
    batch11 = layers.BatchNormalization(momentum=0.01)(conv12)  # affine
    conv12 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu')(batch11)
    out1 = layers.Conv2D(1, 1, strides=(1, 1), padding='same')(conv12)
    print('out1 shape:', out1.shape)
    out2 = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(out1)
    print('out2 shape:', out2.shape)

    model = models.Model(inputs=input1, outputs=out2)
    miou_metric = MeanIoU(num_classes=1)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy', miou_metric.mean_iou, mean_iou2]

    )
    model.summary()
    return model
