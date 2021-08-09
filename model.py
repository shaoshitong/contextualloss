import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Dense, Dropout, Lambda, Concatenate, AvgPool2D, MaxPool2D, \
    Softmax, Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model
import h5py
import numpy as np
import torch
import scipy.io as scio

tf.config.experimental_run_functions_eagerly(True)


class Conttextual_loss(Layer):
    def __init__(self, h=0.1, PONO=False, center_training=True, **kwargs):
        self.h = h
        self.PONO = PONO
        self.center_training = center_training
        super(Conttextual_loss, self).__init__(**kwargs)

    def feature_normalize(self, feature_in, eps=1e-10):
        feature_in_norm = tf.norm(feature_in, 2, 1, keepdims=True) + eps
        feature_in_norm = tf.truediv(feature_in, feature_in_norm)
        return feature_in_norm

    def call(self, input, training=None):
        feature_x, feature_y = input
        feature_x = tf.transpose(feature_x, [0, 3, 1, 2])
        feature_y = tf.transpose(feature_y, [0, 3, 1, 2])
        batch_size = feature_x.shape[0]
        feature_depth = feature_x.shape[1]
        feature_size = feature_x.shape[2]
        ## [N,32,32,5]
        if self.center_training:
            if self.PONO:
                """
                该方法对应的参数为[N,feature,point_x*point_y]
                默认不使用
                """
                feature_x = tf.subtract(feature_x, tf.expand_dims(tf.reduce_mean(feature_y, axis=1), axis=1))
                feature_y = tf.subtract(feature_y, tf.expand_dims(tf.reduce_mean(feature_y, axis=1), axis=1))
            else:
                feature_x = tf.subtract(feature_x, tf.expand_dims(tf.expand_dims(
                    tf.reduce_mean(tf.reshape(feature_y, [batch_size, feature_depth, -1]), axis=-1), axis=-1), axis=-1))
                feature_y = tf.subtract(feature_y, tf.expand_dims(tf.expand_dims(
                    tf.reduce_mean(tf.reshape(feature_y, [batch_size, feature_depth, -1]), axis=-1), axis=-1), axis=-1))
            feature_x = self.feature_normalize(tf.reshape(feature_x, [batch_size, feature_depth,
                                                                      -1]))  # batch_size * feature_depth * feature_size * feature_size
            feature_y = self.feature_normalize(
                tf.reshape(feature_y, [batch_size, feature_depth, -1]))  # batch_size * feature_depth * feature_si
        feature_x_permute = tf.transpose(feature_x, [0, 2, 1])
        d = 1 - tf.matmul(feature_x_permute, feature_y)
        d_norm = d / (tf.reduce_min(d, axis=-1, keepdims=True) + 1e-3)
        w = tf.exp((1 - d_norm) / self.h)
        A_ij = w / tf.reduce_sum(w, axis=-1, keepdims=True)
        CX = tf.reduce_mean(tf.reduce_max(A_ij, axis=-1), axis=1)
        loss = -tf.math.log(CX)
        return loss


class Featureloss(Layer):
    def __init__(self, out_filter, padding_choose, groups=1, pooling_choose='max', strides=(3, 3), kernel_size=(6, 6),
                 use_conv='false', len_conv=3,
                 **kwargs):
        padding_choose: str
        self.out_filter = out_filter
        self.use_conv = use_conv
        self.padding_choose = padding_choose
        self.groups = groups
        self.pooling_choose = pooling_choose
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv_len = len_conv
        super(Featureloss, self).__init__(**kwargs)

    def group_conv(self, x, nb_ig, nb_og, groups=1, feature='x'):

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        in_channels = K.int_shape(x)[channel_axis]  # 计算输入特征图的通道数
        gc_list = []
        for i in range(groups):
            if channel_axis == -1:
                x_group = Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
            else:
                x_group = Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
            if feature == 'x':
                gc_list.append(self.conv_list_x[i](x_group))  # 对每组特征图进行单独卷积
            else:
                gc_list.append(self.conv_list_y[i](x_group))  # 对每组特征图进行单独卷积
        if len(gc_list) == 1:
            return gc_list[-1]
        else:
            return Concatenate(axis=channel_axis)(gc_list)  # 在通道上进行特征图的拼接

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.batch_size = input_shape[0]
        self.feature = input_shape[3]
        self.feature_size1 = input_shape[1]
        self.feature_size2 = input_shape[2]
        self.split_X = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 1})
        self.split_Y = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 1})
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        nb_ig = self.feature // self.groups  # 对输入特征图通道进行分组
        nb_og = self.out_filter // self.groups  # 对输出特征图通道进行分组
        self.conv_list_x = [
            tf.keras.Sequential(
                [tf.keras.layers.Conv2D(filters=nb_og, kernel_size=(3,3), strides=(1,1),
                                        padding='same', use_bias=False) for i in range(self.conv_len)]+[tf.keras.layers.Conv2D(filters=nb_og, kernel_size=self.kernel_size, strides=self.strides,
                                        padding=self.padding_choose, use_bias=False)])
            for _ in range(self.groups)
        ]
        self.conv_list_y = [
            tf.keras.Sequential(
                [tf.keras.layers.Conv2D(filters=nb_og, kernel_size=(3,3), strides=(1,1),
                                        padding='same', use_bias=False) for i in range(self.conv_len)]+[tf.keras.layers.Conv2D(filters=nb_og, kernel_size=self.kernel_size, strides=self.strides,
                                        padding=self.padding_choose, use_bias=False)])
            for _ in range(self.groups)
        ]
        self.conv_x = lambda x: self.group_conv(x, nb_ig=nb_ig, nb_og=nb_og, groups=self.groups, feature='x')
        self.conv_y = lambda x: self.group_conv(x, nb_ig=nb_ig, nb_og=nb_og, groups=self.groups, feature='y')
        if self.pooling_choose == 'max':
            self.pool = MaxPool2D((2, 2), strides=2, padding=self.padding_choose)
        else:
            self.pool = AvgPool2D((2, 2), strides=2, padding=self.padding_choose)
        self.sotfmax = Softmax(axis=-1)
        self.contextualloss1 = Conttextual_loss()
        self.contextualloss2 = Conttextual_loss()
        super(Featureloss, self).build(input_shape)

    def call(self, input, training=None, **kwargs):
        feature_x, feature_y = input
        """
        [N,32,32,5][N,32,32,5]
        """
        if self.use_conv == 'false':
            return self.contextualloss1((feature_x, feature_y))
        feature_x = self.conv_x(feature_x)
        feature_y = self.conv_y(feature_y)
        loss_conv = self.contextualloss2((feature_x, feature_y))
        return loss_conv


def Calculate(feature_size_x, feature_size_y, feature_depth, out_filter, batchsize=64, padding_choose='same',
              use_conv='false'):
    X_IN = Input(shape=(feature_size_x, feature_size_y, feature_depth), batch_size=batchsize)
    Y_IN = Input(shape=(feature_size_x, feature_size_y, feature_depth), batch_size=batchsize)
    loss_layer = Featureloss(out_filter=out_filter, padding_choose=padding_choose, use_conv=use_conv)
    output = loss_layer([X_IN, Y_IN])
    model = Model(inputs=[X_IN, Y_IN], outputs=output)
    return model


model = Calculate(32, 32, 5, 5, batchsize=64, use_conv='true')

data = np.load('new_input.npy')
label = np.load('new_label.npy')
x_train = data[:1024, ...]
y_train = label[:1024, ...]
x_test = data[1024:, ...]
y_test = label[1024:, ...]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
from loss import LossFunction

Loss = LossFunction(learning_rate=10.)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


@tf.function
def train_step(image, labels):
    with tf.GradientTape() as tape:
        output = model(image)
        loss = Loss(labels, output)
        grad_list = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad_list, model.trainable_variables))
        return loss


@tf.function
def test_step(image, labels):
    output = model(image)
    loss = Loss(labels, output)
    return loss


epochs = 80
from tensorflow.keras.metrics import MeanAbsoluteError

train_metric = MeanAbsoluteError()
test_metric = MeanAbsoluteError()

for i in range(epochs):
    train_metric.reset_states()
    test_metric.reset_states()
    for images, labels in train_ds:
        images = [tf.cast(tf.squeeze(_), dtype=tf.float32) for _ in tf.split(images, axis=1, num_or_size_splits=[1, 1])]
        labels = tf.cast(labels, dtype=tf.float32)
        loss_1 = train_step(images, labels)
        train_metric.update_state(loss_1, 0)
    print("the epoch {}'s train loss is {:.3f}".format(i, train_metric.result().numpy().item()))
    for images, labels in test_ds:
        images = [tf.cast(tf.squeeze(_), dtype=tf.float32) for _ in tf.split(images, axis=1, num_or_size_splits=[1, 1])]
        labels = tf.cast(labels, dtype=tf.float32)
        loss_2 = test_step(images, labels)
        test_metric.update_state(loss_2, 0)
    print("the epoch {}'s test loss is {:.3f}".format(i, test_metric.result().numpy().item()))
