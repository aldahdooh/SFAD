#seed_value= 800900

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['PYTHONHASHSEED']=str(seed_value)

import random
#random.seed(seed_value)

import numpy as np
#np.random.seed(seed_value)

import tensorflow.compat.v1 as tf
#tf.set_random_seed(seed_value)

from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# tf.keras.backend.set_session(sess)

import json
import os
import keras
from keras.datasets import cifar10, mnist
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant


def my_generator(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size).next()
        yield [res[0], [res[1], res[1][:, :-1]]]


def my_generator2(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size).next()
        yield [res[0], [res[1], res[1][:, :-1], res[1][:, :-1]]]


def my_generator3(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size).next()
        yield [res[0], [res[1], res[1], res[1], res[1][:, :-1]]]

def my_generator3(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size).next()
        yield [res[0], [res[1], res[1], res[1], res[1][:, :-1]]]


def normalize_mean(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def normalize_linear(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test


def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)

def load_tiny_imagenet_data():
    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_xtrain.bytes', 'rb')
    data_bytes = open_file.read()
    x_train = np.frombuffer(data_bytes, dtype=np.uint8)
    x_train = x_train.reshape(100000,64,64,3)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_ytrain.bytes', 'rb')
    data_bytes = open_file.read()
    y_train = np.frombuffer(data_bytes, dtype=np.uint8)
    y_train = y_train.reshape(100000)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_xtest.bytes', 'rb')
    data_bytes = open_file.read()
    x_test = np.frombuffer(data_bytes, dtype=np.uint8)
    x_test = x_test.reshape(10000,64,64,3)

    open_file = open('/home/aaldahdo/adv_dnn/tiny_imagenet/tiny_ytest.bytes', 'rb')
    data_bytes = open_file.read()
    y_test = np.frombuffer(data_bytes, dtype=np.uint8)
    y_test = y_test.reshape(10000)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)

def load_imagenet2017dev_resnet50v2_data():
    open_file = open('/home/aaldahdo/adv_dnn/imagenet2017final/imagenet_2017_dev_processed_resnet50v2_train.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    x_train = image_data.reshape(4628,224,224,3)
    open_file.close()

    open_file = open('/home/aaldahdo/adv_dnn/imagenet2017final/imagenet_2017_dev_train_labels.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=int)
    y_train = image_data.reshape(4628,1)
    open_file.close()

    open_file = open('/home/aaldahdo/adv_dnn/imagenet2017/imagenet_2017_dev_processed_resnet50v2_test.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    x_test = image_data.reshape(1000,224,224,3)
    open_file.close()

    open_file = open('/home/aaldahdo/adv_dnn/imagenet2017/imagenet_2017_dev_labels_test.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=int)
    y_test = image_data.reshape(1000,1)
    open_file.close()
    
    return (x_train, y_train) , (x_test, y_test)

def load_imagenet2017dev_inceptionv3_data():
    open_file = open('imagenet2017final/imagenet_2017_dev_processed_inceptionv3_train.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    x_train = image_data.reshape(4628,299,299,3)
    open_file.close()

    open_file = open('imagenet2017final/imagenet_2017_dev_labels_train.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    y_train = image_data.reshape(4628,1)
    open_file.close()

    open_file = open('imagenet2017/imagenet_2017_dev_processed_inceptionv3_test.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    x_test = image_data.reshape(1000,299,299,3)
    open_file.close()

    open_file = open('imagenet2017/imagenet_2017_dev_labels_test.bytes', 'rb')
    data_bytes = open_file.read()
    image_data = np.frombuffer(data_bytes, dtype=np.float32)
    y_test = image_data.reshape(1000,1)
    open_file.close()
    return x_train, y_train, x_test, y_test

def toCat_onehot(y_train, y_test, numclasses):
    y_train = keras.utils.to_categorical(y_train, numclasses)
    y_test = keras.utils.to_categorical(y_test, numclasses)

    return y_train, y_test


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def parse_for_shape(layer_file_name):
    layer_file_name = layer_file_name.replace('.bytes', '')
    txt_parts = layer_file_name.split('shape')
    shape_parts = list(filter(None, txt_parts[1].split('_')))
    list_num = []
    for n_str in shape_parts:
        list_num.append(int(n_str))
    shape = tuple(list_num)

    return shape

def get_data_asarray(file_name, data_shape):
    open_file = open(file_name, 'rb')
    data_bytes = open_file.read()
    data_array = np.frombuffer(data_bytes, dtype=np.float32)
    data_array = data_array.reshape(data_shape)
    open_file.close()

    return data_array
