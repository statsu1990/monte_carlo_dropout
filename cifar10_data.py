import keras
from keras.datasets import cifar10
from keras.utils import np_utils

import numpy as np

class Cifar10_Dog_Cat:
    def __init__(self):
        # dog
        self.USE_CIFAR10_LABEL1 = 5
        # cat
        self.USE_CIFAR10_LABEL2 = 3
        
        
        return

    def make_binary_data(self):
        # data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # exract target label (dog and cat)
        target_idx_train = np.where(np.logical_or(np.all(self.y_train==self.USE_CIFAR10_LABEL1, axis=1), np.all(self.y_train==self.USE_CIFAR10_LABEL2, axis=1)))[0]
        self.x_train = self.x_train[target_idx_train]
        self.y_train = self.y_train[target_idx_train]

        target_idx_test = np.where(np.logical_or(np.all(self.y_test==self.USE_CIFAR10_LABEL1, axis=1), np.all(self.y_test==self.USE_CIFAR10_LABEL2, axis=1)))[0]
        self.x_test = self.x_test[target_idx_test]
        self.y_test = self.y_test[target_idx_test]

        # relabel
        self.y_train[self.y_train==self.USE_CIFAR10_LABEL1] = 1
        self.y_train[self.y_train==self.USE_CIFAR10_LABEL2] = 0
        self.y_train = self.y_train.astype(int)

        self.y_test[self.y_test==self.USE_CIFAR10_LABEL1] = 1
        self.y_test[self.y_test==self.USE_CIFAR10_LABEL2] = 0
        self.y_test = self.y_test.astype(int)

        # scaling
        self.x_train = (self.x_train - 122.5) / 255
        self.x_test = (self.x_test - 122.5) / 255

        return

    def make_onehot_data(self):
        # data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # exract target label (dog and cat)
        target_idx_train = np.where(np.logical_or(np.all(self.y_train==self.USE_CIFAR10_LABEL1, axis=1), np.all(self.y_train==self.USE_CIFAR10_LABEL2, axis=1)))[0]
        self.x_train = self.x_train[target_idx_train]
        self.y_train = self.y_train[target_idx_train]

        target_idx_test = np.where(np.logical_or(np.all(self.y_test==self.USE_CIFAR10_LABEL1, axis=1), np.all(self.y_test==self.USE_CIFAR10_LABEL2, axis=1)))[0]
        self.x_test = self.x_test[target_idx_test]
        self.y_test = self.y_test[target_idx_test]

        # relabel
        self.y_train[self.y_train==self.USE_CIFAR10_LABEL1] = 0
        self.y_train[self.y_train==self.USE_CIFAR10_LABEL2] = 1

        self.y_test[self.y_test==self.USE_CIFAR10_LABEL1] = 0
        self.y_test[self.y_test==self.USE_CIFAR10_LABEL2] = 1

        # to one-hot
        self.y_train = np_utils.to_categorical(self.y_train, 2)
        self.y_test = np_utils.to_categorical(self.y_test, 2)

        # scaling
        self.x_train = (self.x_train - 122.5) / 255
        self.x_test = (self.x_test - 122.5) / 255

        return

class Cifar10_1Label:
    def __init__(self, label):
        """
        0 airplane (飛行機)
        1 automobile (自動車)
        2 bird (鳥)
        3 cat (猫)
        4 deer (鹿)
        5 dog (犬)
        6 frog (カエル)
        7 horse (馬)
        8 ship (船)
        9 truck (トラック)
        """
        self.USE_CIFAR10_LABEL = label

        self.__make_data()
        
        return

    def __make_data(self):
        # data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # exract target label
        target_idx_train = np.where(np.all(self.y_train==self.USE_CIFAR10_LABEL, axis=1))[0]
        self.x_train = self.x_train[target_idx_train]
        self.y_train = self.y_train[target_idx_train]

        target_idx_test = np.where(np.all(self.y_test==self.USE_CIFAR10_LABEL, axis=1))[0]
        self.x_test = self.x_test[target_idx_test]
        self.y_test = self.y_test[target_idx_test]

        # scaling
        self.x_train = (self.x_train - 122.5) / 255
        self.x_test = (self.x_test - 122.5) / 255

        return
