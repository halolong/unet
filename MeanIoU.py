import tensorflow as tf
import numpy as np


class MeanIoU(object):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        return tf.py_func(self.np_mean_iou, [y_true, y_pred], tf.float32)

    def np_mean_iou(self, y_true, y_pred):
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        true_positive = np.diag(conf)
        false_postive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive


        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_postive + false_negative)
        iou[np.isnan(iou)] = 0

        return np.mean(iou).astype(np.float32)