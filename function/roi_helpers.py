import numpy as np
import pdb
import math
import copy
from . import deta_generators
import tensorflow as tf
# （w,h,num_anchors）,x_regr(w,h,num_anchor*4)
def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    # TODO:现在的输入形式已经改成了【x1,y1,x2,y2,prob】
    if len(boxes) == 0:
        return []
    # try:
    boxes = np.array(boxes)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # except Exception as e:
    #     print('--------------------------------\n')
    #     print(e)
    #     print('--------------------------------\n')


    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)


    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    # 计算面积
    area = (x2-x1)*(y2-y1)
    # 按概率对boxes进行排序，从小到大的索引

    indexes = np.argsort([i[-1] for i in boxes])
    xxx = len(indexes)
    # print("working okey....")
    while len(indexes) > 0:
        last = len(indexes) - 1
        # 获取目标概率最大的boxes
        i = indexes[last]
        pick.append(i)

        # 找出交集,注意这些地方计算的都是矩阵
        xx1_int = np.maximum(x1[i],x1[indexes[:last]])
        yy1_int = np.maximum(y1[i],y1[indexes[:last]])
        xx2_int = np.minimum(x2[i],x2[indexes[:last]])
        yy2_int = np.minimum(y2[i],y2[indexes[:last]])

        # 最小的宽与高
        ww_int = np.maximum(0,xx2_int-xx1_int)
        hh_int = np.maximum(0,yy2_int-yy1_int)
        # 最小的面积
        area_int = ww_int*hh_int
        # 找到交集
        are_union = area[i] + area[indexes[:last]]-area_int

        # 计算重合率
        overlap = area_int/(are_union+1e-6)

        # 删除交并比大于阈值的部分
        indexes = np.delete(indexes,np.concatenate(([last],np.where(overlap>overlap_thresh)[0])))

        # 只提取前max_boxes个框
        if len(pick) >= max_boxes:
            break
    # print("working ok ...." + str(boxes.shape))
    # try:
    boxes = boxes[pick]
    # print(boxes[0][1])
    # print("working ok ...." + str(boxes.shape) + "")
    return boxes
    # except Exception as e:
    #     print('--------------------------------\n')
    #     print(e)
    #     print('---------------------------------\n')



# rpn网络的数据来对所有的锚框进行修正 以及利用最大值抑制进行框的筛选
def rpn_to_rpi(rpn_layer, regr_layer, cfg, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    regr_layer = regr_layer / cfg.std_scaling
    anchor_sizes = cfg.anchor_box_scales
    anchor_ratios = cfg.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    (rows,cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    A = np.zeros((4,rpn_layer.shape[1],rpn_layer.shape[2],rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            anchor_x = (anchor_size*anchor_ratio[0])/cfg.rpn_stride
            anchor_y = (anchor_size*anchor_ratio[1])/cfg.rpn_stride
            # 当前的回归梯度
            regr = regr_layer[0,:,:,4*curr_layer:4*curr_layer+4]
            regr = np.transpose(regr,(2,0,1))
            # 特征图对应的每个点的坐标

            X,Y = np.meshgrid(np.arange(cols),np.arange(rows))

            # A【4,W,H,9】，每个锚框在特征图上的位置，起点与长宽
            A[0,:,:,curr_layer] = X-anchor_x/2
            A[1,:,:,curr_layer] = Y-anchor_y/2
            A[2,:,:,curr_layer] = anchor_x
            A[3,:,:,curr_layer] = anchor_y
            if use_regr:
                # A【4,W,H,9】,x_regr(num_anchor*4,w,h)
                # 通过梯度修正
                A[:,:,:,curr_layer] = apply_regr_np(A[:,:,:,curr_layer],regr)
                # anchor_x,锚框的宽度如果有小于1的用1来替换
                # anchor_y，锚框的长度如果又小于1的用1来替换，

            A[2:,:,:,curr_layer] = np.maximum(1,A[2,:,:,curr_layer])
            A[3:,:,:,curr_layer] = np.maximum(1,A[3,:,:,curr_layer])
            # 起点加上长宽求出重点的坐标
            A[2,:,:,curr_layer] += A[0,:,:,curr_layer]
            A[3,:,:,curr_layer] += A[1,:,:,curr_layer]

            # 如果起点小于0则将其用0来代替
            # 如果终点大于边界则将其用边界来代替
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])
            # 换到下一个锚框的长宽比
            curr_layer += 1

        all_boxes = np.reshape(A.transpose((0,3,1,2)),(4,-1)).transpose((1,0))
        # print("all boxes shape:"+str(all_boxes.shape))
        # try:
            # all_prop = np.transpose(rpn_layer,(0,3,1,2)).reshape((-1))
        all_prop = tf.transpose(tf.reshape(tf.transpose(rpn_layer,(0,3,1,2)),(1,-1)),(1,0))
        # print("all prop shape:" + str(all_prop.shape))
            # all_prop = rpn_layer.transpose((0,3,1,2)).reshape((-1))

        x1 = all_boxes[:,0]
        y1 = all_boxes[:,1]
        x2 = all_boxes[:,2]
        y2 = all_boxes[:,3]

        ids = np.where((x1-x2>=0)|(y1-y2>=0))

        all_boxes = np.delete(all_boxes,ids,0)
        all_prop = np.delete(all_prop,ids,0)
        # all_prop 为sigmoid分类后的概率值

        all_boxes = np.hstack((all_boxes,all_prop))

            # 非最大值抑制来对候选框进行筛选
        result = non_max_suppression_fast(all_boxes,overlap_thresh = overlap_thresh,
                                              max_boxes = max_boxes)
        # 忽略最后一行的概率
        result = result[:,0:-1]
        return result

# R 经过非最大值抑制后的box
def calc_iou(R,img_data,C,class_mapping):
    bboxes = img_data['bboxes']
    (width,height) = (img_data['width'],img_data['height'])
    (resized_width,resized_height) = deta_generators.get_new_img_size(width,height,C.im_size)
    gta = np.zeros((len(bboxes),4))

    for bbox_num,bbox in enumerate(bboxes):
        gta[bbox_num,0] = int(round(bbox['x1']*(resized_width/float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))
    # 初始化各项参数
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []
    # 对每一个锚框进行循环
    for ix in range(R.shape[0]):
        (x1,y1,x2,y2) = R[ix,:]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # 对于每一个r_box都找出具有最大iou的bbox
        for bbox_num in range(len(bboxes)):
            # 计算iou,计算R里面box的iou与gtbox的iou
            curr_iou = deta_generators.iou([gta[bbox_num,0],gta[bbox_num, 2],gta[bbox_num, 1],gta[bbox_num, 3]],
                                            [x1,y1,x2,y2])
            # 计算每个R——boxes对应的最好的iou的gtbox(其种类即为当前r_box采用的class)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
            # 如果最大的iou不满足分类的最小overlap的要求，直接跳到下一个R_box的循环
        if best_iou < C.classifier_min_overlap:
            continue
            # 否则将该R——box的坐标值存到x_roi当中
            # 将当前的iou存入IOUS当中
        else:
            w = x2-x1
            h = y2-y1
            x_roi.append([x1,y1,w,h])
            IoUs.append(best_iou)
            # 如果分类重叠在0.1~0.5之间，分类为背景
            if C.classifier_min_overlap<=best_iou<C.classifier_max_overlap:
                cls_name = 'bg'
            # 如果大于0.5，将其分类为前景，并计算两个框之间的梯度，使用当前的类别
            elif C.classifier_max_overlap<=best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox,0]+gta[best_bbox,1])/2.0
                cyg = (gta[best_bbox,2]+gta[best_bbox,3])/2.0

                cx = x1+w/2.0
                cy = y1+w/2.0

                tx = (cxg - cx)/float(w)
                ty = (cyg - cy)/float(h)

                tw = np.log((gta[best_bbox,1]-gta[best_bbox,0])/float(w))
                th = np.log((gta[best_bbox,3]-gta[best_bbox,2])/float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # 相当于 one-hot编码，只给相应的类别位置置1
        # 当前类别的编码
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping)*[0]
        class_label[class_num] = 1
        # 将当前的类别编码拷贝一份
        y_class_num.append(copy.deepcopy(class_label))
        # 坐标为
        coords = [0.0]*4*(len(class_mapping)-1)
        labels = [0.0]*4*(len(class_mapping)-1)
        # 如果当前R框不为背景的话，在对应的位置写入坐标与类别编号
        if cls_name != 'bg':
            # 算出该类别对应的坐标位置
            # 猜测：一共10类，每个类有个位置用4个坐标表示，每个类的坐标起始位置为【4*class_num】
            label_pos = 4*class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx,sy*ty,sw*tw,sh*th]
            labels[label_pos:4+label_pos] = [1,1,1,1]

            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi)==0:
        return None,None,None,None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    # bestiou对应锚框坐标，每一个锚框包含类的onehot编码，目标的坐标梯度以及类别编号编码，iou的值
    return np.expand_dims(X,axis=0),np.expand_dims(Y1,axis=0),np.expand_dims(Y2,axis=0),IoUs


def apply_regr_np(X,T):
    try:
        # 锚框在特征图上的位置
        x = X[0,:,:]
        y = X[1,:,:]
        w = X[2,:,:]
        h = X[3,:,:]
        # 回归的值（需要注意在  data_generator中是原图计算，
        # 但是分子分母同时除以16也可忽略不计）
        tx = T[0,:,:]
        ty = T[1,:,:]
        tw = T[2,:,:]
        th = T[3,:,:]
        # 特征图上的锚框中心位置
        cx = x+w/2.
        cy = y+h/2.
        # gtbox的中心点
        cx1 = tx*w + cx
        cy1 = ty*h + cy
        # gtbox的长与宽
        w1 = np.exp(tw.astype(np.float64))*w
        h1 = np.exp(th.astype(np.float64))*h

        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        # 返回gtbox的始终位置
        return np.stack([x1,y1,w1,h1])
    except Exception as e:
        print(e)
        return X


def apply_regr(x,y,w,h,tx,ty,tw,th):
    try:
        cx = x+w/2.
        cy = y+w/2.
        cx1= tx*w + cx
        cy1 = ty*h +cy
        w1 = math.exp(tw)*w
        h1 = math.exp(th)*h
        x1 = cx1-w1/2.
        y1 = cy1-h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1,y1,w1,h1

    except ValueError:
        return x,y,w,h
    except OverflowError:
        return x,y,w,h
    except Exception as e:
        print(e)
        return x,y,w,h



