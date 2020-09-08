import os
import cv2
import xml.etree.ElementTree as Et
import numpy as np

# 从数据集中读取信息
def get_data(input_path):
    all_imgs = []
    class_count = {}
    class_mapping = {}

    visualise = True

    data_paths =[os.path.join(input_path,s) for s in ['VOC2007']]
    print(data_paths)

    print('解析注释文件')

    for data_path in data_paths:
        # xml文件
        annot_path = os.path.join(data_path,'Annotations')
        # 图片文件
        imgs_path = os.path.join(data_path,'JPEGImages')
        # 训练集的目录
        imgsets_path_trainval = os.path.join(data_path,'ImageSets','Main','trainval.txt')
        # 测试集的目录
        imgsets_path_test = os.path.join(data_path,'ImageSets','Main','test.txt')

        trainval_files = []
        test_files = []

        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    #Python strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                    trainval_files.append(line.strip()+'jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip()+'.jpg')
        except Exception as e:
            # 因为voc数据没有test文件
            if data_path[-7:] == 'VOC2007':
                pass
            else:
                print(e)

        if not os.path.exists(annot_path):
            print("路径不存在")

        annots = [os.path.join(annot_path,s) for s in os.listdir(annot_path)]
        idx = 0

        for annot in annots:
            try:
                # 解析xml文件
                idx += 1

                et = Et.parse(annot)
                # 获取根节点
                element = et.getroot()
                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                # 如果图片中包含对象
                if len(element_objs)>0:
                    annotation_data = {'filepath':os.path.join(imgs_path,element_filename),
                                       'width':element_width,'height':element_height,
                                       'bboxes':[]}
                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'
                # 对所有图片里面的对象数量进行计数，以便进行样本均衡
                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if class_name not in class_count:
                        class_count[class_name] = 1
                    else:
                        class_count[class_name] += 1
                    # 如果当前的类别没有在class_mapping,那么当前类的编号为最后一个
                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    # 该对象是不是难以辨别的
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class':class_name,'x1':x1,'x2':x2,'y1':y1,'y2':y2,'difficult':difficulty}
                    )
                all_imgs.append(annotation_data)

                if visualise and idx == 10:

                    img = cv2.imread(annotation_data['filepath'])
                    # print("显示图片"+str(img.shape))
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img,(bbox['x1'],bbox['y1']),(bbox['x2'],bbox[y2]),
                                      (0,0,255))
                    cv2.imshow('img',img)
                    cv2.waitKey(0)
            except Exception as e:
                print(e)
                continue
        return all_imgs,class_count,class_mapping

# main测试
if __name__ == '__main__':
    all_imgs,class_count,class_mapping = \
        get_data("E:\\code\\faster_rnn\\dataset\\VOCtrainval_11-May-2012\\VOCdevkit")
    # 一共20个类，成功
    print(len(class_mapping))