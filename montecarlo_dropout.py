import os

import keras
from keras.layers import Dropout
from keras.models import Model
from keras import backend as K

import numpy as np



# https://qiita.com/NaokiAkai/items/368fc016daa85cf570f0


class MontecarloDropout:
    def __init__(self):
        return

    def build_model(self, model_file_path):
        """
        keras modelからMontecarloDropoutに対応したモデルを作成
        build monte carlo dropout model base on keras_model.
        """

        model = self.__load_model(model_file_path)

        # change dropout layer to dropout layer that can use dropout in inference.
        # ドロップアウト層を推論時にもドロップアウトできるドロップアウト層に変更する。
        for ily, layer in enumerate(model.layers):
            # input layer
            if ily == 0:
                input = layer.input
                h = input
            # is dropout layer ?
            if 'dropout' in layer.name:
                # change dropout layer
                h = Dropout(layer.rate)(h, training=True)
            else:
                h = layer(h)

        self.model = Model(input, h)

        return

    def md_predict(self, xs, sampling_num):
        """
        predict with using monte carlo dropout sampling.
        return prediction average, std

        xs : input sample array. xs = x0, x1, x2, ...
        """

        pre_ys = []
        for ismp in range(sampling_num):
            pre_y = self.model.predict(xs)
            pre_ys.append(pre_y)
        pre_ys = np.array(pre_ys)

        # calculate ave, std
        pre_ave = np.average(pre_ys, axis=0)
        pre_std = np.std(pre_ys, axis=0)

        return pre_ave, pre_std

    def md_freq_dist(self, x, sampling_num):
        """
        frequency distribution of prediction with using monte carlo dropout sampling.
        return predict(x), predict(x), ..., predict(x)
        
        x : one input sample
        
        """

        xs = np.ones((sampling_num, *x.shape)) * x
        pre_y = self.model.predict(xs)

        return pre_y

    def __load_model(self, model_file_path):
        """
        load model .h5 file
        """
        model = keras.models.load_model(model_file_path)
        return model
