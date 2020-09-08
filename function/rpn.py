import tensorflow as tf
from tensorflow_core import keras

from function.vgg16 import nn_base, rpn


class RPN(keras.Model):
    def __init__(self):
        super(RPN,self).__init__()
        self.nn_base = nn_base()
        self.rpn = rpn()
    def call(self,input_image,num_anchors):
        x = self.nn_base(input_image)
        x = self.rpn(x,num_anchors)
        return x


if __name__ == '__main__':
    model = RPN()
    model.build(input_shape=(None, 299, 299, 3))
    model.summary()