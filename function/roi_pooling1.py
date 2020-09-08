import tensorflow as tf
from keras.backend import concatenate

from tensorflow_core import keras
import numpy as np
# 把特征图pooling到特定的大小
from tensorflow_core.python.keras.backend import permute_dimensions


class RoiPoolingConv(tf.keras.layers.Layer):
    def __init__(self,pool_size,num_rois,**kwargs):
        super(RoiPoolingConv,self).__init__()
        # self.pooling_regions = 7
        # self.input_shape = (num_rois,7,7,512)
        self.pool_size = pool_size
        self.num_rois = num_rois

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch,H, W, C = shape
        self.nb_channels = tf.TensorShape(C)
        return None,self.num_rois,self.pool_size,self.nb_channels

    def call(self,x,mask=None):
        # assert(len(x)==2)
        # base_layer
        # input_rois
        img = x[0]
        rois = x[1]
        outputs = []
        input_channel = img.shape[2]
        for roi_idx in range(self.num_rois):
            x = rois[0,roi_idx,0]
            y = rois[0,roi_idx,1]
            w = rois[0,roi_idx,2]
            h = rois[0,roi_idx,3]

            x = tf.cast(x,'int32')
            y = tf.cast(y,'int32')
            w = tf.cast(w,'int32')
            h = tf.cast(h,'int32')

            # rs = tf.image.resize(img[:,y:y+h,x:x+w,:],(self.pool_size,self.pool_size))
            rs = tf.image.resize(img[x:x+w,y:y+h], (self.pool_size, self.pool_size))

            outputs.append(rs)

        final_output = concatenate(outputs,axis=0)
        final_output = tf.reshape(final_output,(1,self.num_rois,self.pool_size,self.pool_size,input_channel))

        final_output = permute_dimensions(final_output,(0,1,2,3,4))

        return final_output

