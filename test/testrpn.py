import tensorflow as tf
from tensorflow_core import keras

from function import vgg16
from function.vgg16 import nn_base, rpn, classfiler


class mymodel1(tf.keras.Model):
    def __init__(self):
        super(mymodel1,self).__init__()
        self.layer1 = nn_base()
        self.layer2 = classfiler(num_rois=4)
    def call(self,img_input,roi_input):
        shared_layer = self.layer1(img_input)
        out_cls,out_reg = self.layer2(shared_layer,roi_input)
        return [out_cls,out_reg]

if __name__ == '__main__':
    layer1 = nn_base()
    layer2 = classfiler(num_rois=4)
    input1 = keras.Input(shape = (None,None,3))
    input2 = keras.Input(shape = (None,4))
    shared_layer = layer1(input1)
    output = layer2(shared_layer,input2)
    model = keras.Model(inputs=[input1,input2],
                    outputs=[output])
    # model = mymodel1()
    # model.build(input_shape = (None,None,None,3))
    # model.build()
    model.load_weights('E:\\code\\faster_rnn\\weight\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    model.layers[0].trainable = False
    model.summary()
    keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
