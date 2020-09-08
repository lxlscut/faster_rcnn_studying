import random

import tensorflow as tf
from tensorflow_core import keras
import numpy as np
from function.GetVoc import get_data
from function.config import Config
from function.deta_generators import get_anchor_gt
from function import vgg16 as nn, roi_helpers
import pickle
import pprint
import cv2

from function.losses import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from function.vgg16 import nn_base, rpn, classfiler


class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1 = nn_base()
        self.layer2 = rpn(num_anchors=9)

    def call(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        return x

class mymodel1(tf.keras.Model):
    def __init__(self):
        super(mymodel1,self).__init__()
        self.layer1 = nn_base()
        self.layer2 = classfiler(num_rois=4)
    def call(self,img_input,roi_input):
        shared_layer = self.layer1(img_input)
        out_cls,out_reg = self.layer2(shared_layer,roi_input)
        return [out_cls,out_reg]

def getdata():
    cfg = Config()
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 32
    all_imgs, class_count, class_mapping = \
        get_data("E:\\code\\faster_rnn\\dataset\\VOCtrainval_11-May-2012\\VOCdevkit")
    if 'bg' not in class_count:
        class_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file,'wb') as config_f:
        pickle.dump(cfg,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))
    print('Training images per class:')
    pprint.pprint(class_count)
    print('Num classes (including bg) = {}'.format(len(class_count)))
    random.shuffle(all_imgs)
    num_imgs = len(all_imgs)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    # print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = get_anchor_gt(all_imgs,class_count,cfg,
                                   nn.get_img_output_length,mode='train')
    return data_gen_train


def getmodel():
    model = mymodel()
    model.build(input_shape = (None,None,None,3))
    model.load_weights('/mnt/sda2/lxl/code/faster_rnn/weight/faster_rcnn.h5',by_name=True)
    model.layers[0].trainable = False
    model.summary()
    # keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    return model

def getmodel1():
    model = mymodel1()
    model.load_weights('/mnt/sda2/lxl/code/faster_rnn/weight/faster_rcnn.h5',by_name=True)
    model.layers[0].trainable = False
    return model

if __name__ == '__main__':

    cfg = Config()
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 32
    all_imgs, class_count, class_mapping = \
        get_data("E:\\code\\faster_rnn\\dataset\\VOCtrainval_11-May-2012\\VOCdevkit")
    if 'bg' not in class_count:
        class_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file,'wb') as config_f:
        pickle.dump(cfg,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))
    print('Training images per class:')
    pprint.pprint(class_count)
    print('Num classes (including bg) = {}'.format(len(class_count)))
    random.shuffle(all_imgs)
    num_imgs = len(all_imgs)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    # print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = get_anchor_gt(all_imgs,class_count,cfg,
                                   nn.get_img_output_length,mode='train')


    # data = getdata()
    X, Y, img_data = next(data_gen_train)
    print(Y[0])
    model = getmodel()
    optimizer = keras.optimizers.Adam(lr=1e-5)
    loss_object = [rpn_loss_cls(num_anchors=9), rpn_loss_regr(num_anchors=9)]
    model.compile(optimizer=optimizer,
                      loss=[rpn_loss_cls(num_anchors=9), rpn_loss_regr(num_anchors=9)]
                  )

    model1 = getmodel1()
    optimizer_classifier = keras.optimizers.Adam(lr=1e-5)
    model1.compile(optimizer = optimizer_classifier,loss = [class_loss_cls,class_loss_regr(len(class_count)-1)])

    rpn_accuracy_rpn_monitor = []

    rpn_accuracy_for_epoch = []


    while True:
        try:
            for i in range(100):
                 X, Y, img_data = next(data_gen_train)
                 # print(X.shape)
                 # print(Y[0].shape)
                 # print(Y[1].shape)
                 loss_rpn = model.fit(X, Y)
                 P_rpn = model.predict_on_batch(X)

                 print(type(P_rpn[0]))

                 result = roi_helpers.rpn_to_rpi(P_rpn[0], P_rpn[1], cfg,use_regr=True,
                                                 overlap_thresh=0.7,
                                                 max_boxes=300)

                 X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)
                 if X2 is None:
                     rpn_accuracy_rpn_monitor.append(0)
                     rpn_accuracy_for_epoch.append(0)
                     continue

                 neg_samples = np.where(Y1[0, :, -1] == 1)
                 pos_samples = np.where(Y1[0, :, -1] == 0)

                 if len(neg_samples) > 0:
                     neg_samples = neg_samples[0]
                 else:
                     neg_samples = []

                 if len(pos_samples) > 0:
                     pos_samples = pos_samples[0]
                 else:
                     pos_samples = []

                 rpn_accuracy_rpn_monitor.append(len(pos_samples))
                 rpn_accuracy_for_epoch.append((len(pos_samples)))

                 if cfg.num_rois > 1:
                     if len(pos_samples) < cfg.num_rois // 2:
                         selected_pos_samples = pos_samples.tolist()
                     else:
                         selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()
                     try:
                         selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                 replace=False).tolist()
                     except:
                         selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                                 replace=True).tolist()

                     sel_samples = selected_pos_samples + selected_neg_samples
                 else:
                     # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                     selected_pos_samples = pos_samples.tolist()
                     selected_neg_samples = neg_samples.tolist()
                     if np.random.randint(0, 2):
                         sel_samples = random.choice(neg_samples)
                     else:
                         sel_samples = random.choice(pos_samples)
                 loss_class = model1.fit([X, X2[:, sel_samples, :]],
                                                              [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                 model1.summary()
                 print(loss_class.shape)
        except Exception as e:
            print(e)
            continue
    print("hahah")