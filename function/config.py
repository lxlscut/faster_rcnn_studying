# 配置文件
class Config:
    def __init__(self):
        # 打印进度条
        self.verbose = True
        # 特征提取网络？？？（实际用的是vgg16）
        # self.network = 'resnet50'
        # setting for data augmentation
        # 图像增强设置
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        # 锚框的scale
        self.anchor_box_scales = [128,256,512]
        # 锚框的比例
        self.anchor_box_ratios = [[1,1],[1,2],[2,1]]
        # 图片的短边长度
        # 为什么要设置短边长度
        self.im_size = 300
        # 图像通道均值
        self.img_channel_mean = [103.99,116.779,123.68]
        self.img_scaling_factor = 1.0
        # roi个数
        self.num_rois = 4

        # 降采样率
        self.rpn_stride = 16
        # 不采用类均衡
        self.balanced_classes = False
        #scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0,8.0,4.0,4.0]
        # rpn重叠
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # roi分类器chongdie
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        # 类的分布，得从文件中读取后才知道
        self.class_mapping =None

        # 保存文件位置
        self.config_save_file = 'config.pickle'

        self.num_epochs = 100
        self.premodel_path = '/mnt/sda2/lxl/code/faster_rnn/weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.model_path = '/mnt/sda2/lxl/code/faster_rnn/weight/faster_rcnn.h5'