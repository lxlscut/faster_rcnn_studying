# # 生成锚框并进行筛选等操作
# from __future__ import absolute_import
# import numpy as np
# import cv2
# import random
# import copy
# import threading
# import itertools
#
# # 求并集
# from function.data_augment import augment
#
#
# def union(au,bu,area_intersection):
#     area_a = (au[2]-au[0])*(au[3]-au[1])
#     area_b = (bu[2]-bu[0])*(bu[3]-bu[1])
#     area_union = area_a+area_b-area_intersection
#     return area_union
#
# # 求交集
# def intersection(ai,bi):
#     x = max(ai[0],bi[0])
#     y = max(ai[1],bi[1])
#
#     w = min(ai[2],bi[2]) - x
#     h = min(ai[3],bi[3]) - y
#
#     if w<0 or h<0:
#         return 0
#     return w*h
#
# # 求交并比
# def iou(a,b):
#     if a[0]>=a[2] or a[1]>a[3] or b[0]>=b[2] or b[1]>= b[3]:
#         return 0.0
#     area_i = intersection(a,b)
#     area_u = union(a,b,area_i)
#     return float(area_i)/float(area_u + 1e-6)
#
# # 获取变换后的图像大小
# def get_new_img_size(width,height,img_min_side = 600):
#     if width<=height:
#         f = float(img_min_side)/width
#         resized_height = int(f*height)
#         resized_width = img_min_side
#     else:
#         f = float(img_min_side)/height
#         resized_width = int(f*width)
#         resized_height = img_min_side
#
#     return resized_width,resized_height
#
# # 抽样,平衡样本数量
# class SampleSelector:
#     def __init__(self,class_count):
#         # 如果该类所包含的对象数量为0，则将其忽略
#         self.classes = [b for b in class_count.keys() if class_count[b]>0]
#         # 迭代器
#         self.class_cycle = itertools.cycle(self.classes)
#         # 获取一个类
#         self.curr_class = next(self.class_cycle)
#
#     def skip_sample_for_banlanced_class(self,img_data):
#         class_in_img = False
#
#         for bbox in img_data['bboxes']:
#             cls_name = bbox['class']
#             if cls_name == self.curr_class:
#                 class_in_img = True
#                 self.curr_class = next(self.class_cycle)
#                 break
#         if class_in_img:
#             return False
#         else:
#             return True
# # todo 计算256个正负样本
# def calc_rpn(C,img_data,width,height,resized_width,resized_height,img_length_calc_function):
#     global best_regr
#     downscale = float(C.rpn_stride)
#     anchor_size = C.anchor_box_scales
#     anchor_ratios = C.anchor_box_ratios
#     # 每个点对应的锚框个数
#     num_anchors = len(anchor_size)*len(anchor_ratios)
#
#     (output_width,output_height) = img_length_calc_function(resized_width,resized_height)
#
#
#
#
#     n_anchratios = len(anchor_ratios)
#
#     # 初始化空对象
#     # 存放所有的锚框对应的overlap
#     y_rpn_overlap = np.zeros((output_height,output_width,num_anchors))
#     # 存放每个锚框是否有效的信息
#     y_is_box_valid = np.zeros((output_height,output_width,num_anchors))
#     # 存放每个锚框的坐标回归值
#     y_rpn_regr = np.zeros((output_height,output_width,num_anchors*4))
#
#     num_bboxes = len(img_data['bboxes'])
#     # 每个bbox对应的anchor
#     num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
#     # 每个bbox对应的最好的anchor的坐标值
#     best_anchor_for_bbox = -1*np.ones((num_bboxes,4)).astype(int)
#     # 每个bbox对应的最好的iou的值
#     best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
#     # 每个bbox对应的最佳的achorbox的坐标值
#     best_x_for_bbox = np.zeros((num_bboxes,4)).astype(int)
#     # 每个bbox对应的最佳的achorbox的梯度值
#     best_dx_for_bbox = np.zeros((num_bboxes,4)).astype(np.float32)
#
#     gta = np.zeros((num_bboxes,4))
#     # 产生一个索引序列
#     # print("开始计算锚框。。。")
#     for bbox_num,bbox in enumerate(img_data['bboxes']):
#         # 框的比例随着图片的resize按比例变小
#         gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
#         gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
#         gta[bbox_num, 2] = bbox['y1'] * (resized_width / float(width))
#         gta[bbox_num, 3] = bbox['y2'] * (resized_width / float(width))
#     # 开始循环
#     # try:
#
#     for anchor_size_idx in range(len(anchor_size)):
#         for anchor_ratio_idx in range(n_anchratios):
#             # anchor_ratios是二维的
#             anchor_x = anchor_size[anchor_size_idx]*anchor_ratios[anchor_ratio_idx][0]
#             anchor_y = anchor_size[anchor_size_idx]*anchor_ratios[anchor_ratio_idx][1]
#
#             for ix in range(output_width):
#                 x1_anc = downscale*(ix+0.5)-anchor_x/2
#
#                 x2_anc = downscale*(ix+0.5)+anchor_x/2
#                 # 锚框越界忽略不计
#                 if x1_anc<0 or x2_anc>resized_width:
#                     continue
#                 for jy in range(output_height):
#                     y1_anc = downscale*(jy+0.5)-anchor_y/2
#                     y2_anc = downscale*(jy+0.5)+anchor_y/2
#                     if y1_anc<0 or y2_anc>resized_height:
#                         continue
#                     # 锚框的类型被固定认为是‘neg’
#                     bbox_type = 'neg'
#
#                     best_iou_for_loc = 0.0
#
#                     # TODO 对于每一个锚框，遍历所有bbox
#                     for bbox_num in range(num_bboxes):
#                         print("真值为：-----------------------------\n")
#                         print("curr_iou 的值为：" + str(num_bboxes))
#                         print("--------------------------------------")
#                         curr_iou = iou([gta[bbox_num, 0],gta[bbox_num, 1],
#                                         gta[bbox_num, 2],gta[bbox_num, 3]],
#                                        [x1_anc,y1_anc,x2_anc,y2_anc])
#                         if curr_iou>0.7:
#                             print("真值为：-----------------------------\n")
#                             print("curr_iou 的值为：" + str(x1_anc))
#                             print("--------------------------------------")
#
#
#                         # 如果当前anchor的iou比阈值大或者大于最佳的iou,就计算梯度备用
#                         if curr_iou>best_iou_for_bbox[bbox_num] or curr_iou>C.rpn_max_overlap:
#                             # 算出中心点
#                             cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
#                             cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
#                             cxa = (x1_anc + x2_anc) / 2.0
#                             cya = (y1_anc + y2_anc) / 2.0
#                             # 计算回归梯度
#                             tx = (cx-cxa)/(x2_anc-x1_anc)
#                             ty = (cy-cya)/(y2_anc-y1_anc)
#                             tw = np.log((gta[bbox_num, 1]-gta[bbox_num, 0])/(x2_anc-x1_anc))
#                             th = np.log((gta[bbox_num, 3]-gta[bbox_num, 2])/(y2_anc-y1_anc))
#                         # 如果当前的bbox不是背景
#                         if img_data['bboxes'][bbox_num]['class'] != 'bg':
#                             # TODO best_iou_for_bbox存放的是每个bbox的最佳iou
#                             if curr_iou>best_iou_for_bbox[bbox_num]:
#                                 # 当前achor对应的坐标（特征图）与anhcor大小
#                                 best_anchor_for_bbox[bbox_num] = [jy,ix,anchor_ratio_idx,anchor_size_idx]
#                                 # 存储当前gtbox的iou
#                                 best_iou_for_bbox = curr_iou
#                                 # 存储当前的anchor坐标（原图）
#                                 best_x_for_bbox[bbox_num,:] = [x1_anc,x2_anc,y1_anc,y2_anc]
#                                 # 存储当前的梯度
#                                 best_dx_for_bbox[bbox_num:] = [tx,ty,tw,th]
#                             if curr_iou>C.rpn_max_overlap:
#
#                                 bbox_type = 'pos'
#                                 # todo 每个bbox对应的positive锚框的个数
#                                 num_anchors_for_bbox[bbox_num]+=1
#                                 # todo 对于每一个锚框的最佳iou与最佳回归参数，每一个锚框起始best_iou_for_loc=0.0
#                                 if curr_iou>best_iou_for_loc:
#                                     best_iou_for_loc = curr_iou
#                                     best_regr = (tx,ty,tw,th)
#
#                             if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
#                                 if bbox_type != 'pos':
#                                     bbox_type = 'neutral'
#
#                     if bbox_type == 'neg':
#                         # output_height, output_width, num_anchors
#                         y_is_box_valid[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size_idx] = 1
#                         y_rpn_overlap[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size_idx] = 0
#                     if bbox_type == 'neutral':
#                         # output_height, output_width, num_anchors
#                         y_is_box_valid[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size_idx] = 0
#                         y_rpn_overlap[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size]=0
#                     if bbox_type == 'pos':
#                         # output_height, output_width, num_anchors
#                         y_is_box_valid[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size_idx] = 1
#                         y_rpn_overlap[jy,ix,anchor_ratio_idx+n_anchratios*anchor_size] = 1
#                         start = 4 * (anchor_ratio_idx+n_anchratios*anchor_size)
#                         y_rpn_regr[jy,ix,start:start+4] = best_regr
#
#
#     # 所有循环完毕，保障每个gtbox至少有一个positive 样本
#     # TODO why??? 猜测是每个gtbox至少有一个正样本才能保证该gtbox内的对象得到训练
#     for idx in range(num_anchors_for_bbox.shape[0]):
#         if num_anchors_for_bbox[idx]==0:
#             # 至少相交
#             if best_anchor_for_bbox[idx,0] == -1:
#                 continue
#             # todo 由于所有的achorbox都没有超过阈值的，所以取最高的
#             y_is_box_valid[best_anchor_for_bbox[idx,0],best_anchor_for_bbox[idx,1],
#             best_anchor_for_bbox[idx,2],best_anchor_for_bbox[idx,3]]
#             y_rpn_overlap[best_anchor_for_bbox[idx,0],best_anchor_for_bbox[idx,1],
#             best_anchor_for_bbox[idx,2],best_anchor_for_bbox[3]]
#             start = 4*(best_anchor_for_bbox[idx,2]+n_anchratios*best_anchor_for_bbox[idx,3])
#             # (output_height, output_width, num_anchors * 4)
#             y_rpn_regr[best_anchor_for_bbox[idx,0],best_anchor_for_bbox[idx,1],
#             start:start+4] = best_dx_for_bbox[idx,:]
#
#     # todo 样本提取完毕，后处理
#     # TODO ???为什么要扩增维度与调换顺序
#     # (x,y,anchors)----->(anchors,x,y)
#     y_rpn_overlap = np.transpose(y_rpn_overlap,(2,0,1))
#     # (achors,x,y)------>(0,anchors,x,y)
#     y_rpn_overlap = np.expand_dims(y_rpn_overlap,axis=0)
#     # (x,y,anchors)------>(0,anchors,x,y)
#     y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
#     y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
#     # (x,y,anchors)------>(0,anchors,x,y)
#     y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
#     y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
#
#     pos_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:]==1,y_is_box_valid[0,:,:,:]==1))
#     neg_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:]==0,y_is_box_valid[0,:,:,:]==1))
#
#     num_pos = len(pos_locs[0])
#
#     print("真值为：-----------------------------\n")
#     print("正样本的值为的值为：" + str(num_pos))
#     print("--------------------------------------")
#
#     # 限制区域共为256个
#     num_regions = 256
#
#     if len(pos_locs[0]) > num_regions/2:
#         val_locs = random.sample(range(len(pos_locs[0])),len(pos_locs[0])-num_regions/2)
#         y_is_box_valid[0,pos_locs[0][val_locs],pos_locs[1][val_locs],pos_locs[2][val_locs]]=0
#         num_pos = num_regions/2
#     if len(neg_locs[0]) + num_pos > num_regions:
#         val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
#         y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
#     #  (output_height, output_width, num_anchors)
#     #  ((output_height,output_width,num_anchors))
#     # 按列拼接
#     y_rpn_cls = np.concatenate([y_is_box_valid,y_rpn_overlap],axis=1)
#     # (output_height, output_width, num_anchors)
#     # (output_height, output_width, num_anchors * 4)
#     y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap,4,axis=1),y_rpn_regr],axis=1)
#
#     return np.copy(y_rpn_cls),np.copy(y_rpn_regr)
#
# # 保证迭代时的线程安全
# class threadsafe_iter:
#     def __init__(self,it):
#         self.it = it
#         self.lock = threading.Lock()
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         with self.lock:
#             return next(self.it)
#
# def threadsafe_generator(f):
#     def g(*a,**kw):
#         return threadsafe_iter(f(*a,**kw))
#     return g
#
#
# def get_anchor_gt(all_img_data,class_count,C,img_length_calc_function,mode='train'):
#     # 类均衡
#     sample_selector = SampleSelector(class_count)
#     # print("准备进入get_anchor_gt函数。。。")
#     while True:
#         if mode == 'train':
#             random.shuffle(all_img_data)
#
#         for img_data in all_img_data:
#             try:
#                 if C.banlanced_classes and sample_selector.skip_sample_for_banlanced_class(img_data):
#                     continue
#                 if mode == 'train':
#                     img_data_aug,x_img = augment(img_data,C,augment=True)
#                 else:
#                     img_data_aug, x_img = augment(img_data, C, augment=False)
#
#
#
#                 (width,height) = (img_data_aug['width'],img_data_aug['height'])
#                 (rows,cols,_) = x_img.shape
#
#                 assert cols == width
#                 assert rows == height
#                 # 图像大小resize
#                 (resized_width,resize_height) = get_new_img_size(width,height,C.im_size)
#
#                 x_img = cv2.resize(x_img,(resized_width,resize_height),interpolation=cv2.INTER_CUBIC)
#                 try:
#                     # print("准备进入calc_rpn函数。。。")
#                     y_rpn_cls,y_rpn_reg = calc_rpn(C,img_data_aug,width,height,
#                                                     resized_width,resize_height,img_length_calc_function)
#
#
#                 except Exception as e:
#                     # print("calc_rpn出现错误。。。")
#                     # print(e)
#                     continue
#                 # cv2读取的为BGR，转为RGB
#                 x_img = x_img[:,:,(2,1,0)]
#                 x_img = x_img.astype(np.float32)
#
#                 x_img[:,:,0] -= C.img_channel_mean[0]
#                 x_img[:, :, 1] -= C.img_channel_mean[1]
#                 x_img[:, :, 2] -= C.img_channel_mean[2]
#                 x_img /= C.img_scaling_factor
#                 # (w,h,(r,g,b))---->>>((r,g,b),w,h)
#                 x_img = np.transpose(x_img,(2,0,1))
#                 # ((r,g,b),w,h)---->>>(0,(r,g,b),w,h)
#                 x_img = np.expand_dims(x_img,axis=0)
#                 # 回归梯度除以一个规整因子
#                 y_rpn_reg[:,y_rpn_reg.shape[0]//2,:,:] *= C.std_scaling
#                 # (0,w,h,(R,G,B))
#                 x_img = np.transpose(x_img,(0,2,3,1))
#                 y_rpn_cls = np.transpose(y_rpn_cls,(0,2,3,1))
#                 y_rpn_reg = np.transpose(y_rpn_reg,(0,2,3,1))
#
#                 yield np.copy(x_img),[np.copy(y_rpn_cls),np.copy(y_rpn_reg)],img_data_aug
#             except Exception as e:
#                 # print(e)
#                 # print("get anchor 出现错误")
#                 continue


from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools
import tensorflow as tf

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height

class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        # 所具有的类别
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        # 迭代器
        self.class_cycle = itertools.cycle(self.classes)
        # 获取一个类
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:
            # 获取bbox的类别
            cls_name = bbox['class']
            #
            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break
        # 该类里面有图片？？？
        if class_in_img:
            return False
        else:
            return True


# RPN 计算，
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    # 降采样率
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    # 每个锚点对应的锚框个数
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture
    # 传入resize后的图片，求出输出特征图的大小
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    # 初始化空的对象
    # 存放每个锚框的overlap信息
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    # 存放每个锚框是否有效的信息
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    # 存放每个锚框的坐标信息
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
    # 手动标注的目标框的个数
    num_bboxes = len(img_data['bboxes'])
    # 存放每个bbox对应的锚框信息
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    # 存放每个bbox对应的最佳iou
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    # ?????
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    # convert the label bbox to resized image, just change the ratio according to resize/original_size
    # gta 载入最佳手动标注的目标框的坐标信息
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    # iterate anchor size and ratio, get all possiable RPNs
    #
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            # 对九种锚框尺寸进行循环
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            # TODO: Important part, we got final feature map output_w and output_h
            # then we reflect back every anchor positions in the original image
            for ix in range(output_width):
                # x-coordinates of the current anchor box
                # 计算出锚框的x左右的坐标
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2
                # 如果锚框超出边界则忽略不计
                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                #  计算锚框的y轴信息，
                for jy in range(output_height):
                    # 如果y坐标超出边界，也可以忽略不计
                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue
                    # 锚框的类型默认为 negtive
                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    # 最佳的iou初始为0
                    best_iou_for_loc = 0.0
                    # 对于每一个锚框，都要对所有的手动标注框进行适配
                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        # 算出当前的iou
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        # 如果当前的iou大于对应gtbox的最大iou或者大于iou的阈值
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0
                            # 计算梯度值
                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        # 如果当前的gtbox的标签不为背景
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            # 如果当前的iou大于当前gtbox有的最佳iou
                            # 将当前的anchors信息存起来
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                # 存储当前anchor对应的特征图坐标，anchor的比例，大小
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                # 存储当前的iou
                                best_iou_for_bbox[bbox_num] = curr_iou
                                # 存储当前的anchor坐标
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                # 存储当前的梯度
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            # 如果当前的iou大于设置的最大的overlap,为正样本
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                # gtbox对应的anchors数量+1
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                # 如果当前的iou大于best_iou_for_loc
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            # 如果iou范围在最大与最小之间
                            # 设置器类型为中立的
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    # box type has neg,pos, neutral
                    # neg: the box is backgronud
                    # pos: the box has RPN
                    # neutral: normal box with a RPN
                    # 由于初始的bbox_type为'neg'所以如果不满足，则进行下面的操作
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region
    # 以下确保每个gtbox至少会有一个正样本，原本只有大于阈值才有，现在只要iou里面最高的那个即可
    # 对所有的gtbox进行遍历
    for idx in range(num_anchors_for_bbox.shape[0]):
        # 如果该bbox对应的anchors为0
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            # 如果直接没有相交直接下一个
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]
    #

    # 存放每个锚框的overlap信息
    # y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    # 存放每个锚框是否有效的信息
    # y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    # 存放每个锚框的坐标信息
    # y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    # 添加一个0位置的维度可以方便统计正负样本的数量
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    # 获取正负样本的位置
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])
    # print("---------------->>>>" + str(num_pos))
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    # 总共选出的正负样本总数为256个
    num_regions = 256
    # 如果正样本的数量超过了一般，则使得超过的部分为无效样本
    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
    # 是否有效以及是否正负样本
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    # 按列合并
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    # y_rpn_cls为样本的有效性以及正负，y_rpn_regr为样本正负以及坐标信息
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)



# 线程管理
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


# 获取anchor框

# data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,
# 											   K.image_dim_ordering(), mode='train')

# all_image:所有的图片【图片名，图片大小，图片所包含的对象，所包含的对象的坐标位置，是训练还是验证】
# class_count:每个种类所含的图片数量
# C为配置在文件的对象
# img_length_calc_function：通过resnet获取的输出尺寸
# backend
def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, mode='train'):
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    # 创建SampleSelector对象
    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    # 是否使用图像增强
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)
                # 获取图像的宽与高
                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                # 对图像大小进行规范化
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # resize the image so that smalles side is length = 600px
                try:
                    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
                    print(e)
                    print(width,height,resized_height,resized_width)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

                try:
                    # 获取anchor的获取分类信息，回归信息，正负样本个128个
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
                                                     img_length_calc_function)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image
                # ？？？？做什么的--->将图像由BGR模式转换为RGB模式
                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                # 像素值减去均值，0中值
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor
                # print(x_img.dtype)
                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)
                # ？？？什么用
                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling


                x_img = np.transpose(x_img, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue







