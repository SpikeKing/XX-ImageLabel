#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/11
"""
import cv2
import os

from root_dir import ROOT_DIR, IMG_DATA
from utils.xml_processor import read_anno_xml


def draw_img(image, boxes):
    img = cv2.imread(image)
    img_width = img.shape[0] / 100
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        i1_pt1 = (int(x_min), int(y_min))
        i1_pt2 = (int(x_max), int(y_max))
        cv2.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, thickness=img_width, color=(255, 0, 255))

    cv2.imshow('Image', img)
    cv2.imwrite('./img_346.bbox.jpg', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # logAll/儿童_0.jpg 1566,344,1695,519,0
    img_path = os.path.join(IMG_DATA, 'logAll-0717/ErTong_6.jpg')
    box_path = os.path.join(IMG_DATA, 'logAll-0717/ErTong_6.xml')
    # print(img_path)
    # 183,279,624,613
    # boxes = [(836, 504, 1339, 1081)]
    draw_img(img_path, read_anno_xml(box_path))
