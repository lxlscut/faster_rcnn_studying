# 数据增强
import cv2
import numpy as np
import copy

from function.GetVoc import get_data
from function.config import Config


def augment(img_data,config,augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    # 拷贝源文件且不改变源文件的内容
    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    # 如果使用图像增强的话
    try:
        if augment:
            # (w,h,c)
            rows,cols = img.shape[:2]
            # 如果使用了水平翻转，可能采取翻转策略
            if config.use_horizontal_flips and np.random.randint(0,2)==0:
                img = cv2.flip(img,1)
                # 翻转后标注框也需要翻转
                for bbox in img_data_aug['bboxes']:
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    bbox['y1'] = rows - y2
                    bbox['y2'] = rows - y1
            # 竖直翻转
            if config.use_vertical_flips and np.random.randint(0,2)==0:
                img = cv2.flip(img,0)
                for bbox in img_data_aug['bboxes']:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    bbox['x1'] = cols-x2
                    bbox['x2'] = cols-x1

            if config.rot_90:
                # np.random.choice为选出一个子数组，加0方为数值
                angle = np.random.choice([0,90,180,270],1)[0]
                if angle == 270:
                    img = np.transpose(img,(1,0,2))
                    img = cv2.flip(img,0)
                elif angle == 180:
                    img = cv2.flip(img,-1)
                elif angle == 90:
                    img = np.transpose(img,(1,0,2))
                elif angle == 0:
                    pass
                for bbox in img_data_aug['bboxes']:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    if angle == 270:
                        bbox['x1'] = y1
                        bbox['x2'] = y2
                        bbox['y1'] = cols - x2
                        bbox['y2'] = cols - x1
                    elif angle == 180:
                        bbox['x2'] = cols - x1
                        bbox['x1'] = cols - x2
                        bbox['y2'] = rows - y1
                        bbox['y1'] = rows - y2
                    elif angle == 90:
                        bbox['x1'] = rows - y2
                        bbox['x2'] = rows - y1
                        bbox['y1'] = x1
                        bbox['y2'] = x2
                    elif angle == 0:
                        pass
    except Exception as e:
        print(e)
    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug,img

# 测试可以实现图像增强
# if __name__ == '__main__':
#     all_imgs,class_count,class_mapping = \
#         get_data("E:\\code\\faster_rnn\\dataset\\VOCtrainval_11-May-2012\\VOCdevkit")
#     c = Config()
#     print(len(all_imgs))
#     img_data_aug,img = augment(all_imgs[10], c, augment=True)
#     cv2.rectangle(img, (10, 10), (40, 40), (0, 255, 0), 2)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
#     print()