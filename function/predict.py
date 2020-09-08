from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow_core import keras

from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras import Model
import argparse
import os

# 返回resize后的图像与比例
from function import roi_helpers
from function.vgg import nn_base, rpn,classifier
import function.vgg as nn
from function.visualize import draw_boxes_and_label_on_image_cv2


def format_img_size(img,cfg):
    img_min_side = float(cfg.im_size)
    (height,width,_) = img.shape

    if width<=height:
        ratio = img_min_side/width
        new_height = int(ratio*height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio*width)
        new_height = int(img_min_side)

    img = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_CUBIC)
    return img,ratio

# 对通道均值标准化，归一化
def format_img_channels(img,cfg):
    img = img[:,:,(2,1,0)]
    img = img.astype(np.float32)
    img[:,:,0] -= cfg.img_channel_mean[0]
    img[:,:,1] -= cfg.img_channel_mean[1]
    img[:,:,2] -= cfg.img_channel_mean[2]

    img /= cfg.img_scaling_factor
    img = np.transpose(img,(2,1,0))
    img = np.expand_dims(img,axis=0)
    return img

# 原始图片-->resized--->去中值标准化
def format_img(img,c):
    img,ratio = format_img_size(img,c)
    img = format_img_channels(img,c)
    return img,ratio

# 原图标注的框转换为将要处理的框的坐标
def get_real_coordinates(ratio,x1,x2,y1,y2):
    real_x1 = int(round(x1//ratio))
    real_y1 = int(round(y1//ratio))
    real_x2 = int(round(x2//ratio))
    real_y2 = int(round(y2//ratio))

    return real_x1,real_y1,real_x2,real_y2

def predict_single_image(img_path,model_rpn,model_classifier_only,cfg,class_mapping):
    st = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print('reading images failed')
        exit(0)
    X,ratio = format_img(img,cfg)
    X = np.transpose(X,(0,2,3,1))
    [Y1,Y2,F] = model_rpn.predict(X)

    result = roi_helpers.rpn_to_rpi(Y1,Y2,cfg,overlap_thresh=0.7)
    # 从 （x1,y1,x2,y2,p） to (x,y,w,h,p)
    result[:,2]-=result[:,0]
    result[:,3]-result[:,1]
    bbox_threshold = 0.8
    # dict()函数创建一个字典
    # 在推荐区域使用空间金字塔
    boxes = dict()
    # result【num,5】
    for jk in range(result.shape[0]//cfg.num_rios + 1):
        # rois[0,4,5]
        rois = np.expand_dims(result[cfg.num_rios*jk:cfg.num_rios*(jk+1),:],axis=0)
        if rois.shape[1] == 0:
            break
        # 最后一次循环,如果数据不够的话pad上去一个
        if jk == result.shape[0]//cfg.num_rios:
            curr_shape = rois.shape
            target_shape = (curr_shape[0],cfg.num_rios,curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:,:curr_shape[1],:] = rois
            rois_padded[0,curr_shape[1]:,:] = rois[0,0,:]
            rois = rois_padded
        # pcls[4,21],p_regr[4,80]
        [p_cls,p_regr] = model_classifier_only.predict([F,rois])

        for ii in range(p_cls.shape[1]):
            # 那个概率最大则证明是哪个类，最大值不超过阈值进行下一次循环
            if np.max(p_cls[0,ii,:])<bbox_threshold or np.argmax(p_cls[0,ii,:])==(p_cls.shape[2]-1):
                continue
            # 类别编号为最大概率的编号
            cls_num = np.argmax(p_cls[0,ii,:])

            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            # 对坐标进行回归校正
            (x,y,w,h) = rois[0,ii,:]
            try:
                # 分类编号与回归编号是一一对应的
                (tx,ty,tw,th) = p_regr[0,ii,4*cls_num:4*(cls_num+1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x,y,w,h = roi_helpers.apply_regr(x,y,w,h,tx,ty,tw,th)
            except Exception as e:
                print(e)
                pass
            # 将每个类别对应的框的信息存储起来，每个类别有多少个框都存储起来，
            boxes[cls_num].append([cfg.rpn_stride*x,cfg.rpn_stride*y,
                                   cfg.rpn_stride*(x+w),cfg.rpn_stride*(y+h),np.max(p_cls[0,ii,:])])
    # 对每个类别的所有的框做最大值抑制
    for cls_num,box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box,overlap_thresh=0.5)
        boxes[cls_num] = boxes_nms
        print(class_mapping[cls_num]+":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            print('{} prob: {}'.format(b[0: 4], b[-1]))
    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    print('Elapsed time = {}'.format(time.time() - st))
    cv2.imshow('image', img)
    result_path = './results_images/{}.png'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img)
    cv2.waitKey(0)

def predict(agrs_):
    path = agrs_.path
    # 训练时保存的配置文件
    with open('config.pickle','rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v:k for k,v in class_mapping.items()}
    input_shape_img = (None,None,3)
    input_shape_features = (None,None,1024)

    img_input = Input(shape = input_shape_img)
    roi_input = Input(shape = (cfg.num_rois,4))
    feature_map_input = Input(shape = input_shape_features)
    shared_layers = nn_base(img_input,trainable=True)

    num_anchors = len(cfg.anchor_box_scales)*len(cfg.anchor_box_ratios)
    rpn_layers = rpn(shared_layers,num_anchors)
    classifier = nn.classifier(feature_map_input,roi_input,cfg.num_rois,nb_classes = len(class_mapping),
                            trainable = True)
    model_rpn = Model(img_input,rpn_layers)
    model_classifier = Model([feature_map_input,roi_input],classifier)

    print('loading weight from{}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path,by_name=True)
    model_classifier.load_weights(cfg.model_path,by_name = True)
    model_rpn.compile(optimizer = 'sgd',loss = 'mse')
    model_classifier.compile(optimizer = 'sgd',loss = 'mse')
    if os.path.isdir(path):
        for idx,img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp','.jpeg','.jpg','.png','.tif','.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path,img_name),model_rpn,
                                 model_classifier,cfg,class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path,model_rpn,model_classifier,cfg,class_mapping)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path','-p',default='images/000010.png',help='images path')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    predict(args)
        



