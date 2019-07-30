import os

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.utils import plot_model
import numpy as np



class ClassifierCnn:
    """
    classify dog and cat in cifar10
    """
    def __init__(self):
        return

    def built_model(self, dropout_rate, l2, input_shape=None):
        # constants
        self.DROPOUT_RATE = dropout_rate
        self.L2 = l2

        if input_shape is None:
            # assume cifar10 image
            input_shape = (32, 32, 3)

        # model structure
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', 
                         kernel_regularizer=regularizers.l2(self.L2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('relu'))
        model.add(Dropout(self.DROPOUT_RATE))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('relu'))
        model.add(Dropout(self.DROPOUT_RATE))
        model.add(Dense(1, kernel_regularizer=regularizers.l2(self.L2)))
        model.add(Activation('sigmoid'))

        self.model = model

        self.model.summary()

        return

    def train_model(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        # compile
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # datagen
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.1,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(x_train)

        # fit
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            validation_data=(x_test, y_test),
                            )

        # score
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])

        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return

    def save_model(self, save_file_name):
        """
        save model
        """
        # dir
        dir_name = os.path.dirname(save_file_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        # save model
        self.model.save(save_file_name)
        print('Saved trained model at %s ' % save_file_name)
        
        # visualize
        plot_model(self.model, to_file=os.path.join(dir_name, 'model_structure.png'), show_shapes=True, show_layer_names=False)

        """
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # model file
        model_path = os.path.join(save_dir, 'trained_model.h5')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        """
        
        return

    def load_model(self, model_file_path):
        """
        load model .h5 file
        """
        self.model = keras.models.load_model(model_file_path)
        return







