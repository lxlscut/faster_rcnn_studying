"""
this code will train on kitti data set
"""
from __future__ import division
from tensorflow import keras
import random
import pprint
import sys
import time
import numpy as np
import pickle
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow_core.python.keras.layers import Input
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.utils import generic_utils
import os
import tensorflow as tf
from function import config, roi_helpers
from function.GetVoc import get_data
from function.deta_generators import get_anchor_gt
from function.losses import rpn_loss_cls, class_loss_cls, rpn_loss_regr, class_loss_regr
from function.roi_helpers import rpn_to_rpi
from function.vgg import nn_base, rpn, classifier, get_img_output_length


def train_kitti():
    # config for data argument
    cfg = config.Config()

    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 4
    cfg.base_net_weights = '/mnt/sda2/lxl/code/faster_rnn/weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # TODO: the only file should to be change for other data to train

    # 从文件中文件具体信息，种类数与种类信息
    # all_image:所有的图片【图片名，图片大小，图片所包含的对象，所包含的对象的坐标位置，是训练还是验证】
    # class_count 为每个class的图片数量
    # class_mapping为一共有多少个class
    all_images, classes_count, class_mapping = get_data("/mnt/sda2/lxl/code/faster_rnn/dataset/VOCtrainval_06-Nov-2007/VOCdevkit")
    # 如果背景的数量为零
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    cfg.class_mapping = class_mapping
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cfg.config_save_file))

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_imgs = [s for s in all_images if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_images if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    # 获取对应的anchor_box坐标与回归信息，256个正负样本
    data_gen_train = get_anchor_gt(train_imgs, classes_count, cfg, get_img_output_length,
                                                    mode='train')
    data_gen_val = get_anchor_gt(val_imgs, classes_count, cfg, get_img_output_length,
                                                  mode='val')


    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    # 共享层
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    # 定义rpn层，每个锚点锚框的数量
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    # rpn的输入为共享层，以及锚框的数量，（分别做anchors卷积然后分类然后  anchors*4 回归）
    # 输出为[x_class, x_regr, base_layers]
    # x_class的大小为（w,h,num_anchors）,x_regr(w,h,num_anchor*4)
    rpn1 = rpn(shared_layers, num_anchors)
    #

    classifier1 = classifier(shared_layers, roi_input, cfg.num_rois, nb_classes=len(classes_count), trainable=True)
    # rpn模型的预测输出为目标的类别以及坐标x_class, x_regr
    model_rpn = Model(img_input, rpn1[:2])
    model_classifier = Model([img_input, roi_input], classifier1)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    # [out_class, out_regr]
    model_all = Model([img_input, roi_input], rpn1[:2] + classifier1)

    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.premodel_path, by_name=True)
        model_classifier.load_weights(cfg.premodel_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    optimizer_classifier = tf.keras.optimizers.Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[class_loss_cls, class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 1000
    num_epochs = int(cfg.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True

    for epoch_num in range(num_epochs):
        # 进度条
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')
                # np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
                try:
                    X, Y, img_data = next(data_gen_train)
                except Exception as e:
                    print(e)

                ###########这里训练的数据为图片，y为分类与回归情况###########
                ###########通过特有的损失函数来使得rpn网络得到训练###########
                loss_rpn = model_rpn.train_on_batch(X, Y)
                #########rpn网络预测的输出[x_class, x_regr]
                # x_class的大小为（w,h,num_anchors）,x_regr(w,h,num_anchor*4)
                P_rpn = model_rpn.predict_on_batch(X)
                # print(type(P_rpn[0]))
                result = rpn_to_rpi(P_rpn[0], P_rpn[1], cfg,use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # 对各个框框进行分类  坐标/类别
                # Y1为类别的one-hot编码
                # Y2为回归的类别与坐标
                # X2为所有满足最小classifier_min_overlap的[x1, y1, w, h]
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(result, img_data, cfg, class_mapping)
                # np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs



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

                # Y1为类别的one-hot编码
                # Y2为回归的类别与坐标
                # X2为所有满足最小classifier_min_overlap的[x1, y1, w, h]
                # X为原图
                # [img_input, roi_input]
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)
                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
    print('Training complete, exiting.')


if __name__ == '__main__':
    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_kitti()
