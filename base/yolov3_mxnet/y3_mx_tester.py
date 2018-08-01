#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import os
import sys

import cv2
import copy
import numpy as np
from PIL import Image
from mxnet import nd

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from base.yolov3_mxnet.darknet import DarkNet
from base.yolov3_mxnet.dir_consts import MODEL_DATA, CONFIGS_DATA
from base.yolov3_mxnet.y3_utils import try_gpu, load_classes, prep_image, predict_transform, write_results
from root_dir import IMG_DATA
from utils.alg_utils import bb_intersection_over_union
from utils.dtc_utils import read_anno_xml, draw_boxes, draw_boxes_simple, filter_sbox, make_line_colors, \
    format_img_and_anno
from utils.log_utils import print_info
from utils.project_utils import safe_div, mkdir_if_not_exist


class YoloVerification(object):
    def __init__(self, img_folder, out_folder):
        self.img_folder = img_folder
        self.out_folder = out_folder

        mkdir_if_not_exist(out_f)  # 创建文件夹

        self.params_path = os.path.join(MODEL_DATA, 'yolov3.weights')  # YOLO v3 权重文件
        self.classes_path = os.path.join(CONFIGS_DATA, 'coco.names')  # 类别文件
        self.targets_path = os.path.join(CONFIGS_DATA, 'traffic.names')

        self.classes_name = load_classes(self.classes_path)  # 加载类别目录
        self.num_classes = len(self.classes_name)  # 类别数

        self.targets_name = load_classes(self.targets_path)

        self.anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                 (59, 119), (116, 90), (156, 198), (373, 326)])  # anchors

        self.confidence = 0.50  # 置信度
        self.nms_thresh = 0.20  # NMS阈值
        self.input_dim = 416  # YOLOv3的检测尺寸

        gpu = '1'  # GPU
        gpu = [int(x) for x in gpu.replace(" ", "").split(",")]
        self.ctx = try_gpu(gpu)[0]  # 选择ctx

        self.net = self.load_model()  # 加载网络

    def load_model(self):
        net = DarkNet(input_dim=self.input_dim, num_classes=self.num_classes)  # 基础网络DarkNet
        net.initialize(ctx=self.ctx)

        if self.params_path.endswith(".params"):  # 加载模型
            net.load_params(self.params_path)
        elif self.params_path.endswith(".weights"):
            tmp_batch = nd.uniform(shape=(1, 3, self.input_dim, self.input_dim), ctx=self.ctx)
            net(tmp_batch)
            net.load_weights(self.params_path, fine_tune=False)
        else:
            print("params {} load error!".format(self.params_path))
            exit()
        print_info("加载参数: {}".format(self.params_path))
        net.hybridize()

        return net

    def detect_image_facets(self, img_path):
        image_data = cv2.imread(img_path)  # 读取图片数据
        image_reform = prep_image(image_data, self.input_dim)
        image_arr = nd.array([image_reform], ctx=self.ctx)
        prediction = predict_transform(self.net(image_arr), self.input_dim, self.anchors)
        pred_res = write_results(prediction, self.num_classes, confidence=self.confidence, nms_conf=self.nms_thresh)
        boxes, scores, classes = self.generate_bboxes(image_data, pred_res, input_dim=self.input_dim)
        return boxes, scores, classes

    def generate_bboxes(self, img_data, bbox_raw, input_dim):
        """
        将原始数据转换为框
        :param img_data: 图片数据
        :param bbox_raw: 原始数据
        :param input_dim: 图片维度
        :return: 框, 置信度, 类别
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
            boxes_.append(bbox[1:5])
            scores_.append(bbox[6])  # 概率
            classes_.append(bbox[7])

        return boxes_, scores_, classes_

    def verify_model(self):
        """
        处理标记文件夹
        :param img_folder: 图片文件夹
        :param out_folder:
        :return:
        """
        img_dict = format_img_and_anno(self.img_folder)

        res_list = []
        for count, img_name in enumerate(img_dict):
            print_info('-' * 50)
            print_info('图片: {}'.format(img_name))
            (img_p, anno_p) = img_dict[img_name]
            res_dict = self.detect_img(img_p, anno_p, self.out_folder)
            res_list.append(res_dict)
            # if count == 10:
            #     break
        print_info('-' * 50)

        for target_name in (['all'] + self.targets_name):
            if target_name in ['truck', 'bus']:
                continue
            ap, ar, count = 0, 0, 0
            for pr_dict in res_list:
                if target_name in pr_dict:
                    tp, tr = pr_dict[target_name]
                    ap += tp
                    ar += tr
                    count += 1
            mAp = safe_div(ap, count)
            mAr = safe_div(ar, count)

            print_info('类: {} P: {:.4f} %, R: {:.4f} %'.format(target_name, mAp * 100, mAr * 100))

    def detect_img(self, img_path, xml_path, out_folder=None):

        color_list_1 = make_line_colors(n_color=20, alpha=0.6, bias=1.0)
        color_list_2 = make_line_colors(n_color=20, alpha=1.0, bias=0.8)

        car_list = ['truck', 'bus', 'car']

        # img_path = os.path.join(IMG_DATA, 'jiaotong-0727', '6emekgFq0neQv8ELXmCq5aQnL3Zz.jpg')
        # xml_path = os.path.join(IMG_DATA, 'jiaotong-0727', '6emekgFq0neQv8ELXmCq5aQnL3Zz.xml')
        img_data = Image.open(img_path)

        boxes, scores, classes_no = self.detect_image_facets(img_path)  # 检测图片
        boxes = self.reform_boxes(boxes)
        classes = [self.classes_name[int(i)] for i in classes_no]  # 将classes的no转换为name
        boxes, scores, classes = filter_sbox((img_data.size[0], img_data.size[1]), (boxes, scores, classes))
        boxes, scores, classes = self.keep_classes(self.targets_name, boxes, scores, classes)
        classes = self.merge_classes_name(car_list, 'car', classes)
        print_info('检测数: {} - {}'.format(len(boxes), classes))

        t_boxes, t_classes = read_anno_xml(xml_path)
        t_scores = ['T' for i in range(len(t_boxes))]
        t_boxes, t_scores, t_classes = \
            filter_sbox((img_data.size[0], img_data.size[1]), (t_boxes, t_scores, t_classes))
        t_boxes, t_scores, t_classes = self.keep_classes(self.targets_name, t_boxes, t_scores, t_classes)
        t_classes = self.merge_classes_name(car_list, 'car', t_classes)
        print_info('真值数: {}'.format(len(t_boxes), t_classes))

        uni_classes = sorted(list(set(classes) | set(t_classes)))
        color_dict_1 = dict(zip(uni_classes, color_list_1[:len(uni_classes)]))
        color_dict_2 = dict(zip(uni_classes, color_list_2[:len(uni_classes)]))

        img_true = draw_boxes_simple(img_data, t_boxes, t_scores, t_classes, color_dict_2)
        # img_true.show()
        img_pred = draw_boxes_simple(img_data, boxes, scores, classes, color_dict_1, is_alpha=True)
        # img_pred.show()
        if out_folder:
            img_name = img_path.split('/')[-1]
            out_img = os.path.join(out_folder, img_name + '.d.png')
            img_pred.save(out_img)  # 存储

        res_dict = dict()

        # print('all')
        print_info('交通工具')
        ap, ar = self.iou_of_boxes(t_boxes, boxes)
        res_dict['all'] = (ap, ar)
        print_info('')

        for class_name in uni_classes:
            print_info('类别: {}'.format(class_name))
            sub_t_boxes, sub_t_classes = self.filter_class(class_name, t_boxes, t_classes)
            sub_boxes, sub_classes = self.filter_class(class_name, boxes, classes)
            ap, ar = self.iou_of_boxes(sub_t_boxes, sub_boxes)
            res_dict[class_name] = (ap, ar)

        return res_dict

    @staticmethod
    def merge_classes_name(name_list, target_name, classes_name):
        res_list = []
        for name in classes_name:
            if name in name_list:
                res_list.append(target_name)
            else:
                res_list.append(name)
        return res_list

    # def detect_img(self, img_path, anno_path, out_folder=None):
    #     """
    #     检测图片，与标注对比
    #     :param yolo: YOLO
    #     :param img_path: 图片路径
    #     :param anno_path: 标注路径
    #     :param out_folder: 存储错误图像
    #     :return:
    #     """
    #
    #     boxes, scores, classes = self.detect_image_facets(img_path)
    #     boxes = self.reform_boxes(boxes)
    #     print_info('检测: {}'.format(boxes))
    #
    #     anno_boxes, name_list = read_anno_xml(anno_path)
    #     anno_boxes = self.reform_boxes(anno_boxes)
    #     print_info('真值: {}'.format(anno_boxes))
    #
    #     precision, recall = self.iou_of_boxes(boxes, anno_boxes)
    #
    #     img_data = Image.open(img_path)
    #     if out_folder and (precision != 1.0 or recall != 1.0):
    #         img_data = draw_boxes(img_data, boxes, scores, classes, [(255, 0, 0)], self.classes_name)
    #         img_data = draw_boxes(img_data, anno_boxes,
    #                               [1.0 for i in range(len(anno_boxes))],
    #                               [0 for i in range(len(anno_boxes))],
    #                               [(128, 128, 255)],
    #                               self.classes_name)
    #         # 图片的缩略图
    #         # size = 1024, 1024
    #         # img_data.thumbnail(size, Image.ANTIALIAS)
    #         img_name = img_path.split('/')[-1]
    #         out_img = os.path.join(out_folder, img_name + '.d.png')
    #         img_data.save(out_img)  # 存储
    #         # img_data.show()  # 显示
    #     return img_path, precision, recall

    @staticmethod
    def keep_classes(class_names, boxes, scores, classes):
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
    def filter_class(clazz_name, boxes, classes):
        t_boxes, t_classes = [], []
        for box, clazz in zip(boxes, classes):
            if clazz_name == clazz:
                t_boxes.append(box)
                t_classes.append(clazz)
        return t_boxes, t_classes

    @staticmethod
    def iou_of_boxes(boxes1, boxes2):
        """
        框的IOU
        :param boxes1_: 真值
        :param boxes2_: 预测
        :return: 精准和召回
        """
        res_list = []

        boxes1_ = copy.deepcopy(boxes1)
        boxes2_ = copy.deepcopy(boxes2)

        t1, t2 = len(boxes1_), len(boxes2_)

        for box1 in boxes1_:
            final_iou = 0
            final_box = None
            for box2 in boxes2_:
                iou = bb_intersection_over_union(box1, box2)
                if iou > final_iou:
                    final_iou = iou
                    final_box = box2
            if final_iou > 0.5:
                res_list.append((box1, final_box, final_iou))
                boxes2_.remove(final_box)

        tr = len(res_list)

        if tr == 0:  # 没有检测源, 则认为正确
            if t1 == 0 and t2 != 0:
                recall, precision = 1.0, 0.0
            elif t1 != 0 and t2 == 0:
                recall, precision = 0.0, 1.0
            elif t1 != 0 and t2 != 0:
                recall, precision = 0.0, 0.0
            else:
                recall, precision = 1.0, 1.0
        else:
            recall = safe_div(tr, t1)  # 召回率
            precision = safe_div(tr, t2)  # 精准率

        print_info('精准: {:.4f}, 召回: {:.4f}, 匹配结果: {}'.format(precision, recall, res_list))

        return [precision, recall]

    @staticmethod
    def reform_boxes(boxes):
        r_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            r_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        return r_boxes


if __name__ == '__main__':
    img_f = os.path.join(IMG_DATA, 'jiaotong-0727')
    out_f = os.path.join(IMG_DATA, 'jiaotong-0727-xxx')
    yv = YoloVerification(img_f, out_f)
    yv.verify_model()
    # img_path = os.path.join(IMG_DATA, 'jiaotong-0727', 'K58KG2vtR6G4j2j6PLhDPn1EdEwg.jpg')
    # xml_path = os.path.join(IMG_DATA, 'jiaotong-0727', 'K58KG2vtR6G4j2j6PLhDPn1EdEwg.xml')
    # yv.detect_img(img_path, xml_path, out_f)
