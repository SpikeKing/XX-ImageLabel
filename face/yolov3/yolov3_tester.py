#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import os

from PIL import Image

from face.yolov3.yolov3_alg import YoloV3

from root_dir import IMG_DATA

from utils.log_utils import print_info
from utils.xml_processor import read_anno_xml


def detect_img():
    img_path = os.path.join(IMG_DATA, 'logAll-0717', 'ErTong_40.jpg')
    img_data = Image.open(img_path)
    yolo = YoloV3()
    boxes, scores, classes = yolo.detect_image_facets(img_data)
    boxes = reform_boxes(boxes)
    print_info('检测: {}'.format(boxes))

    anno_path = os.path.join(IMG_DATA, 'logAll-0717', 'ErTong_40.xml')
    anno_boxes = read_anno_xml(anno_path)
    anno_boxes = reform_boxes2(anno_boxes)
    print_info('真值: {}'.format(anno_boxes))

    img_data = yolo.draw_boxes(img_data, boxes, scores, classes)
    size = 1024, 1024
    img_data.thumbnail(size, Image.ANTIALIAS)
    img_data.show()


def iou_of_boxes(boxes1, boxes2):


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
    detect_img()
