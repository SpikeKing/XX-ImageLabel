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
from utils.alg_utils import bb_intersection_over_union
from utils.log_utils import print_info
from utils.project_utils import safe_div, traverse_dir_files, mkdir_if_not_exist
from utils.xml_processor import read_anno_xml


def verify_yolo(img_folder, out_folder):
    """
    处理标记文件夹
    :param img_folder: 图片文件夹
    :param out_folder:
    :return:
    """
    yolo = YoloV3()  # yolo算法类

    img_dict = format_img_and_anno(img_folder)
    _, file_names = traverse_dir_files(out_folder)

    res_dict = dict()
    for count, img_name in enumerate(img_dict):
        print_info('-' * 50)
        print_info('图片: {}'.format(img_name))
        (img_p, anno_p) = img_dict[img_name]
        _, precision, recall = detect_img(yolo, img_p, anno_p)
        res_dict[img_name] = (precision, recall)
        if count == 20:
            break

    ap, ar = 0, 0
    for name in res_dict.keys():
        precision, recall = res_dict[name]
        ap += precision
        ar += recall
    mAp = safe_div(ap, len(res_dict.keys()))
    mAr = safe_div(ar, len(res_dict.keys()))
    print_info('平均精准率: {:.4f} %, 平均召回率: {:.4f} %'.format(mAp * 100, mAr * 100))


def format_img_and_anno(img_folder):
    """
    格式化输出。图片和标注文件夹
    :param img_folder: 图片文件夹
    :return:
    """
    file_paths, file_names = traverse_dir_files(img_folder)
    img_dict = dict()  # 将标注和图片路径，生成一个字典
    for file_path, file_name in zip(file_paths, file_names):
        if file_name.endswith('.jpg'):
            name = file_name.strip('.jpg')
            if name not in img_dict:
                img_dict[name] = (None, None)
            (img_p, anno_p) = img_dict[name]
            img_dict[name] = (file_path, anno_p)

        if file_name.endswith('.xml'):
            name = file_name.strip('.xml')
            if name not in img_dict:
                img_dict[name] = (None, None)
            (img_p, anno_p) = img_dict[name]
            img_dict[name] = (img_p, file_path)

    print_info('图片数: {}'.format(len(img_dict.keys())))
    return img_dict


def detect_img(yolo, img_path, anno_path, is_show=False):
    # img_path = os.path.join(IMG_DATA, 'logAll-0717', 'ErTong_40.jpg')
    img_data = Image.open(img_path)
    # yolo = YoloV3()
    boxes, scores, classes = yolo.detect_image_facets(img_data)
    boxes = reform_boxes(boxes)
    print_info('检测: {}'.format(boxes))

    # anno_path = os.path.join(IMG_DATA, 'logAll-0717', 'ErTong_40.xml')
    anno_boxes = read_anno_xml(anno_path)
    anno_boxes = reform_boxes2(anno_boxes)
    print_info('真值: {}'.format(anno_boxes))

    precision, recall = iou_of_boxes(boxes, anno_boxes)

    if is_show:
        img_data = yolo.draw_boxes(img_data, boxes, scores, classes)
        img_data = yolo.draw_boxes(img_data, anno_boxes,
                                   [1.0 for i in range(len(anno_boxes))],
                                   [0 for i in range(len(anno_boxes))],
                                   [(128, 128, 255)])

        size = 1024, 1024
        img_data.thumbnail(size, Image.ANTIALIAS)
        img_data.show()
    return img_path, precision, recall


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
    t1 = len(boxes1)
    t2 = len(boxes2)
    tr = len(res_list)
    recall = safe_div(tr, t1)  # 召回率
    precision = safe_div(tr, t2)  # 精准率
    print_info('精准: {:.4f}, 召回: {:.4f}, 匹配结果: {}'.format(precision, recall, res_list))
    return [precision, recall]


def reform_boxes(boxes):
    r_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        r_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return r_boxes


def reform_boxes2(boxes):
    r_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        r_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return r_boxes


if __name__ == '__main__':
    img_f = os.path.join(IMG_DATA, 'logAll-0717')
    out_f = os.path.join(IMG_DATA, 'logAll-0717-xxx')
    mkdir_if_not_exist(out_f)
    verify_yolo(img_f, out_f)
