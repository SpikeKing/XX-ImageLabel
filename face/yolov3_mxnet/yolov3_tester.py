#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import os

import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from PIL import Image

from face.yolov3.yolov3_alg import YoloV3
from root_dir import IMG_DATA

from utils.dtc_utils import read_anno_xml, format_img_and_anno
from utils.alg_utils import bb_intersection_over_union
from utils.log_utils import print_info
from utils.project_utils import safe_div, mkdir_if_not_exist


class YoloVerification(object):
    def __init__(self, img_folder, out_folder):
        self.img_folder = img_folder
        self.out_folder = out_folder

        mkdir_if_not_exist(out_f)

        self.model_path = 'model_data/ep074-loss26.535-val_loss27.370.h5',
        self.classes_path = 'configs/wider_classes.txt',
        self.anchors_path = 'configs/yolo_anchors.txt'

    def verify_model(self):
        """
        处理标记文件夹
        :param img_folder: 图片文件夹
        :param out_folder:
        :return:
        """
        yolo = YoloV3(model_path=self.model_path,
                      classes_path=self.classes_path,
                      anchors_path=self.anchors_path)  # yolo算法类

        img_dict = format_img_and_anno(self.img_folder)

        res_dict = dict()
        for count, img_name in enumerate(img_dict):
            print_info('-' * 50)
            print_info('图片: {}'.format(img_name))
            (img_p, anno_p) = img_dict[img_name]
            _, precision, recall = self.detect_img(yolo, img_p, anno_p, self.out_folder)
            res_dict[img_name] = (precision, recall)
            # if count == 20:
            #     break

        ap, ar = 0, 0
        for name in res_dict.keys():
            precision, recall = res_dict[name]
            ap += precision
            ar += recall

        mAp = safe_div(ap, len(res_dict.keys()))
        mAr = safe_div(ar, len(res_dict.keys()))
        print_info('平均精准率: {:.4f} %, 平均召回率: {:.4f} %'.format(mAp * 100, mAr * 100))

    def detect_img(self, yolo, img_path, anno_path, out_folder=None):
        """
        检测图片，与标注对比
        :param yolo: YOLO
        :param img_path: 图片路径
        :param anno_path: 标注路径
        :param out_folder: 存储错误图像
        :return:
        """
        img_data = Image.open(img_path)
        boxes, scores, classes = yolo.detect_image_facets(img_data)
        boxes = self.reform_boxes(boxes)
        print_info('检测: {}'.format(boxes))

        anno_boxes = read_anno_xml(anno_path)
        anno_boxes = self.reform_boxes2(anno_boxes)
        print_info('真值: {}'.format(anno_boxes))

        precision, recall = self.iou_of_boxes(boxes, anno_boxes)

        if out_folder and (precision != 1.0 or recall != 1.0):
            img_data = yolo.draw_boxes(img_data, boxes, scores, classes, [(255, 0, 0)])
            img_data = yolo.draw_boxes(img_data, anno_boxes,
                                       [1.0 for i in range(len(anno_boxes))],
                                       [0 for i in range(len(anno_boxes))],
                                       [(128, 128, 255)])
            # 图片的缩略图
            # size = 1024, 1024
            # img_data.thumbnail(size, Image.ANTIALIAS)
            img_name = img_path.split('/')[-1]
            out_img = os.path.join(out_folder, img_name + '.d.png')
            img_data.save(out_img)  # 存储
            # img_data.show()  # 显示
        return img_path, precision, recall

    @staticmethod
    def iou_of_boxes(boxes1, boxes2):
        res_list = []
        for box1 in boxes1:
            final_iou = 0
            final_box = None
            for box2 in boxes2:
                iou = bb_intersection_over_union(box1, box2)
                if iou > final_iou:
                    final_iou = iou
                    final_box = box2
            if final_iou > 0.5:
                res_list.append((box1, final_box, final_iou))

        t1, t2, tr = len(boxes1), len(boxes2), len(res_list)

        if tr == 0:  # 没有检测源, 则认为正确
            return [1.0, 1.0]

        recall = safe_div(tr, t1)  # 召回率
        precision = safe_div(tr, t2)  # 精准率
        print_info('精准: {:.4f}, 召回: {:.4f}, 匹配结果: {}'.format(precision, recall, res_list))
        return [precision, recall]

    @staticmethod
    def reform_boxes(boxes):
        r_boxes = []
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            r_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        return r_boxes

    @staticmethod
    def reform_boxes2(boxes):
        r_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            r_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        return r_boxes


if __name__ == '__main__':
    img_f = os.path.join(IMG_DATA, 'logAll-0717')
    out_f = os.path.join(IMG_DATA, 'logAll-0717-xxx')
    yv = YoloVerification(img_f, out_f)
    yv.verify_model()
