#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/1

YOLO v3的检测核心类
"""
import collections
import os

import cv2
import numpy as np
from PIL import Image
from mxnet import nd

from base.yolov3_mxnet.darknet import DarkNet
from base.yolov3_mxnet.dir_consts import MODEL_DATA, CONFIGS
from base.yolov3_mxnet.y3_utils import load_classes, try_gpu, reform_img, predict_transform, filter_results
from root_dir import IMG_DATA
from utils.alg_utils import bb_intersection_over_union
from utils.dtc_utils import draw_boxes_simple, make_line_colors, draw_title
from utils.log_utils import print_info
from utils.project_utils import sort_dict_by_value


class Y3Model(object):

    def __init__(self):
        self.params_path = os.path.join(MODEL_DATA, 'yolov3.weights')  # YOLO v3 参数 for COCO
        self.classes_path = os.path.join(CONFIGS, 'coco.names')  # 类别文件

        self.classes_name = load_classes(self.classes_path)  # 加载类别目录
        self.num_classes = len(self.classes_name)  # 类别数

        self.anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                 (59, 119), (116, 90), (156, 198), (373, 326)])  # anchors

        self.confidence = 0.50  # 框置信度
        self.nms_thresh = 0.20  # NMS阈值
        self.prob_thresh = 0.50  # 类别置信度

        self.input_dim = 416  # YOLOv3的检测尺寸

        gpu = '1'  # GPU
        gpu = [int(x) for x in gpu.replace(" ", "").split(",")]
        self.ctx = try_gpu(gpu)[0]  # 选择ctx

        self.net = self.__load_model()  # 加载网络

    def __load_model(self):
        """
        加载网络模型
        :return: 网络
        """
        net = DarkNet(input_dim=self.input_dim, num_classes=self.num_classes)  # 基础网络 DarkNet
        net.initialize(ctx=self.ctx)  # 网络环境 cpu or gpu

        print_info("模型: {}".format(self.params_path))

        if self.params_path.endswith(".params"):  # 加载参数
            net.load_params(self.params_path)
        elif self.params_path.endswith(".weights"):  # 加载模型
            tmp_batch = nd.uniform(shape=(1, 3, self.input_dim, self.input_dim), ctx=self.ctx)
            net(tmp_batch)
            net.load_weights(self.params_path, fine_tune=False)
        else:
            raise Exception('模型错误')  # 抛出异常

        net.hybridize()  # 编译和优化网络

        return net

    def __detect_img_facets(self, img_data):
        """
        检测图片细节
        :param img_data: 图片数据
        :return: 框, 置信度, 类别编号
        """
        img_reformed = reform_img(img_data, self.input_dim)  # 将图片转换(416, 416), 值为(0~1)
        img_arr = nd.array([img_reformed], ctx=self.ctx)  # 预测批次

        # 将网络输出,转换为图片原始尺寸
        pred_raw = predict_transform(self.net(img_arr), self.input_dim, self.anchors)

        # 框置信度的过滤和NMS
        pred_res = filter_results(pred_raw, self.num_classes, confidence=self.confidence, nms_conf=self.nms_thresh)

        # 恢复原始框的大小和位置
        boxes, scores, classes = self.__generate_bboxes(
            img_data, pred_res, input_dim=self.input_dim, prob_thresh=self.prob_thresh)

        return boxes, scores, classes

    @staticmethod
    def __generate_bboxes(img_data, bbox_raw, input_dim, prob_thresh):
        """
        将原始数据转换为框
        :param img_data: 图片数据
        :param bbox_raw: 原始数据
        :param input_dim: 图片维度
        :param prob_thresh: 类别置信度
        :return: 框, 置信度, 类别编号
        """
        img_size = nd.array([(img_data.shape[1], img_data.shape[0])])  # 图片高和宽
        img_size_all = nd.tile(img_size, 2)  # 重复两次, 生成4个坐标，对应(xmin, ymin, xmax, ymax)
        img_size_list = img_size_all[bbox_raw[:, 0], :]  # 重复生成多行，每个框对应1行

        sf = nd.min(input_dim / img_size_list, axis=1).reshape((-1, 1))  # 压缩比例
        sw_dim_list = sf * img_size_list[:, 0].reshape((-1, 1))  # 宽比例
        sh_dim_list = sf * img_size_list[:, 1].reshape((-1, 1))  # 高比例

        bbox_raw[:, [1, 3]] -= (input_dim - sw_dim_list) / 2  # 1,3宽
        bbox_raw[:, [2, 4]] -= (input_dim - sh_dim_list) / 2  # 2,4高
        bbox_raw[:, 1:5] /= sf  # 等比除法，还原

        for i in range(bbox_raw.shape[0]):
            bbox_raw[i, [1, 3]] = nd.clip(bbox_raw[i, [1, 3]], a_min=0.0, a_max=img_size_list[i][0].asscalar())
            bbox_raw[i, [2, 4]] = nd.clip(bbox_raw[i, [2, 4]], a_min=0.0, a_max=img_size_list[i][1].asscalar())

        bbox_raw = bbox_raw.asnumpy()

        boxes_, scores_, classes_ = [], [], []

        for bbox in bbox_raw:  # [图片索引, xmin, ymin, xmax, ymax, conf, prob, class_id]
            if bbox[6] >= prob_thresh:  # 过滤小于类别置信度的框
                boxes_.append(bbox[1:5])
                scores_.append(bbox[6])  # 概率
                classes_.append(bbox[7])

        return boxes_, scores_, classes_

    @staticmethod
    def __keep_classes(class_names, boxes, scores, classes):
        t_boxes, t_scores, t_classes = [], [], []
        for box, score, clazz in zip(boxes, scores, classes):
            if clazz in class_names:
                t_boxes.append(box)
                t_scores.append(score)
                t_classes.append(clazz)
            else:
                continue
        return t_boxes, t_scores, t_classes

    @staticmethod
    def __map_classes(merge_dict, classes_name):
        res_list = []
        for name in classes_name:
            if name in merge_dict:
                res_list.append(merge_dict[name])
            else:
                res_list.append(name)
        return res_list

    @staticmethod
    def __ratio_of_boxes(img_area, box):
        img_area = float(img_area)
        box = [float(x) for x in box]
        top, left, bottom, right = box
        return abs(top - bottom) * abs(left - right) / img_area

    @staticmethod
    def __nms_boxes(boxes, scores, classes, nms=0.2):
        scores, boxes, classes = zip(*sorted(zip(scores, boxes, classes), reverse=True))
        mask_bb = [True for i in range(len(scores))]

        for i in range(len(scores)):
            box1, score1, class1 = boxes[i], scores[i], classes[i]
            for j in range(i + 1, len(scores)):
                if not mask_bb[j]:
                    continue
                box2, score2, class2 = boxes[j], scores[j], classes[j]
                iou = bb_intersection_over_union(box1, box2)
                if iou >= nms:
                    mask_bb[j] = False
                # print(i, j, iou)

        t_boxes, t_scores, t_classes = [], [], []
        for bb, box, score, clazz in zip(mask_bb, boxes, scores, classes):
            if bb:
                t_boxes.append(box)
                t_scores.append(score)
                t_classes.append(clazz)

        return t_boxes, t_scores, t_classes

    def detect_img(self, img_path, is_img=False):
        img_data = cv2.imread(img_path)  # 读取图片数据
        boxes, scores, classes_no = self.__detect_img_facets(img_data=img_data)  # 检测图片
        classes = [self.classes_name[int(i)] for i in classes_no]  # 将classes的no转换为name

        targets_path = os.path.join(CONFIGS, 'traffic.names')  # 交通工具类别
        target_names = load_classes(targets_path)  # 工具名称

        boxes, scores, classes = self.__keep_classes(target_names, boxes, scores, classes)  # 过滤其他类别

        if boxes and scores and classes_no:
            return self.process_traffic(boxes=boxes, scores=scores, classes=classes, img_path=img_path,
                                        target_names=target_names, is_img=is_img)
        else:
            return {}, None

    def process_traffic(self, boxes, scores, classes, img_path, target_names, is_img):
        merge_dict = {'truck': 'car', 'bus': 'car', 'car': 'car', 'motorbike': 'bicycle'}
        classes = self.__map_classes(merge_dict, classes)  # 合并类别
        boxes, scores, classes = self.__nms_boxes(boxes, scores, classes)

        tag_list = list(set([merge_dict[n] if n in merge_dict else n for n in target_names]))

        tag_ratio_res = collections.defaultdict(float)
        img_data = Image.open(img_path)
        img_area = img_data.size[0] * img_data.size[1]

        for box, clazz in zip(boxes, classes):
            ratio = self.__ratio_of_boxes(img_area, box)
            tag_ratio_res[clazz] += ratio

        if is_img:
            str_list = []
            for tag, value in sort_dict_by_value(tag_ratio_res):
                ratio = tag_ratio_res[tag] if tag in tag_ratio_res else 0
                data_str = '{}:{:.2f}%'.format(tag, ratio * 100)
                str_list.append(data_str)
            label_str = ', '.join(str_list)

            color_list = make_line_colors(n_color=len(tag_list))
            color_dict = dict(zip(tag_list, color_list[:len(tag_list)]))

            img_box = draw_boxes_simple(img_data, boxes, scores, classes, color_dict)
            img_box = draw_title(img_data, label_str)

            return tag_ratio_res, img_box

        return tag_ratio_res, None


def test_of_y3model():
    img_path = os.path.join(IMG_DATA, 'jiaotong-0727', 'nG16LBFdlGmkLNnjQOS2lk1lgDav.jpg')
    ym = Y3Model()
    tag_ratio_res, img_box = ym.detect_img(img_path)


if __name__ == '__main__':
    test_of_y3model()
