# 定义基础网络，rpn网络，与最终的分类回归网络

from __future__ import print_function
from __future__ import absolute_import
import cv2
import numpy as np
import warnings

import tensorflow as tf
from tensorflow_core import keras
from tensorflow_core.python.keras.layers.wrappers import TimeDistributed

from function.roi_pooling1 import RoiPoolingConv


def get_img_output_length(width,height):
    def get_output_length(input_length):
        return input_length//16
    return get_output_length(width),get_output_length(height)

# 特征提取层，采用vgg16来实现
# 特征提取层，采用vgg16来实现
class nn_base(tf.keras.layers.Layer):
    def __init__(self):
        super(nn_base,self).__init__()
        self.conv1_1 = keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same',name='block1_conv1')
        self.conv1_2 = keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same',name='block1_conv2')
        self.pool1 = keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block1_pool')
        self.conv2_1 = keras.layers.Conv2D(128,(3,3),activation = 'relu',padding = 'same',name='block2_conv1')
        self.conv2_2 = keras.layers.Conv2D(128,(3,3),activation = 'relu',padding = 'same',name='block2_conv2')
        self.pool2 = keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')
        self.conv3_1 = keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name='block3_conv1')
        self.conv3_2 = keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name='block3_conv2')
        self.conv3_3 = keras.layers.Conv2D(256,(3,3),activation = 'relu',padding = 'same',name='block3_conv3')
        self.pool3 = keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')
        self.conv4_1 =keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block4_conv1')
        self.conv4_2 = keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block4_conv2')
        self.conv4_3 = keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block4_conv3')
        self.pool4 = keras.layers.MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')
        self.conv5_1 =keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block5_conv1')
        self.conv5_2 = keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block5_conv2')
        self.conv5_3 = keras.layers.Conv2D(512,(3,3),activation = 'relu',padding = 'same',name='block5_conv3')
        # self.conv1 = keras.layers.Conv2D(filter_num = 64,kernel_size = (3,3),padding = 'same',activation='relu')
    def call(self,input,**kwargs):
        # block1
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        x = self.pool1(x)
        # block2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        # block3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        # block4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        # block5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        output = self.conv5_3(x)
        return x
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch,H, W, C = shape
        output_shape = tf.TensorShape([batch,H//16,W//16,512])
        return output_shape

# rpn网络
class rpn(keras.layers.Layer):
    def __init__(self,num_anchors):
        super(rpn,self).__init__()
        self.anchors = num_anchors
        # 为什么256？？？？
        self.conv1 = keras.layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',
                                kernel_initializer='normal',name = 'rpn_conv1')
        self.cls = keras.layers.Conv2D(self.anchors,(1,1),activation = 'sigmoid',
                                      kernel_initializer = 'uniform',name = 'rpn_out_class')
        self.reg = keras.layers.Conv2D(self.anchors*4,(1,1),activation='linear',
                                     kernel_initializer='zero',name = 'rpn_out_reg')
    def call(self,base_layers,**kwargs):

        x = self.conv1(base_layers)
        x_class = self.cls(x)
        x_regr = self.reg(x)
        return x_class,x_regr
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch, H, W, C = shape
        output_shape1 = tf.TensorShape([batch,H,W,self.anchors])
        output_shape2 = tf.TensorShape([batch, H, W, self.anchors*4])
        return (output_shape1,output_shape2)


# 最终分类网络
class classfiler(keras.layers.Layer):
    def __init__(self,num_rois,nb_classes = 21):
        super(classfiler,self).__init__()
        self.pooling_regions = 7
        # self.input_shape = (num_rois,7,7,512)
        self.out_roi_pool = RoiPoolingConv(pool_size=self.pooling_regions, num_rois=num_rois)
        self.out1 = TimeDistributed(keras.layers.Flatten(name = 'flatten'))
        self.out2 = TimeDistributed(keras.layers.Dense(4096,activation = 'relu',name = 'fc1'))
        self.out3 = TimeDistributed(keras.layers.Dense(4096,activation = 'relu',name = 'fc2'))
        self.out_cls = TimeDistributed(keras.layers.Dense(nb_classes,activation = 'softmax',
                                                          kernel_initializer='zero'),
                                                          name='dense_class_{}'.format(nb_classes))
        self.out_reg = TimeDistributed(keras.layers.Dense(4*(nb_classes-1),activation = 'linear',
                                                          kernel_initializer='zero'),
                                       name='dense_regress_{}'.format(nb_classes)
                                       )
    def call(self,base_layers,input_rois,**kwargs):
        # 待写
        out0 = self.out_roi_pool(base_layers,input_rois)
        out1 = self.out1(out0)
        out2 = self.out2(out1)
        out3 = self.out3(out2)

        out_cls = self.out_cls(out3)
        out_reg = self.out_reg(out3)

        return [out_cls,out_reg]


class base_model(tf.keras.Model):
    def __init__(self):
        super(base_model,self).__init__()
        self.nn = nn_base()
    def call(self,input,**kwargs):
        x = self.nn(input)
        return x
if __name__ == '__main__':
    print(" ")
