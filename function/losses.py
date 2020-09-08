import tensorflow as tf
from tensorflow_core import keras
import numpy as np
from tensorflow_core.python.keras import backend as K
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4



def rpn_loss_regr(num_anchors):

	def rpn_loss_regr_fixed_num(y_true, y_pred):
         try:
             # print("rpn_loss_regr y_true的值："+str(y_true.shape))
             # print("rpn_loss_regr y_pred：" + str(y_pred.shape))
             x = y_true[:, :, :, 4 * num_anchors:] - y_pred
             x_abs = K.abs(x)
             x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
             tf.less_equal
             tf.abs
             return lambda_rpn_regr * K.sum(
                 y_true[:, :, :, :4 * num_anchors]
                 * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
         except Exception as e:
             print("regr错误："+str(e))
	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):

         try:
            # print("y_true的值：" + str(y_true.shape))
               # print("y_pred：" + str(y_pred.shape))
            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] *
                                                K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
         except Exception as e:
             print("cls错误："+str(e))
	return rpn_loss_cls_fixed_num



def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):

         try:
             x = y_true[:, :, 4 * num_classes:] - y_pred
         except Exception as e:
             print('-----------------------------------------------\n')
             print('reg错误：' + str(e))
             print('-------------------------------------------------\n')
         x_abs = K.abs(x)
         x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        # 会为0的错误
         return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])

	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    try:
	    return lambda_cls_class * K.mean(keras.losses.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    except Exception as e:
        print('----------------------------------------------------\n')
        print(e)
        print('----------------------------------------------------\n')
