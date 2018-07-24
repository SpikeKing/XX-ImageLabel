#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import os

import cv2

from root_dir import IMG_DATA
from utils.log_utils import print_info
from utils.project_utils import *
from utils.xml_processor import read_anno_xml





def draw_img(image, boxes, out_folder, file_names):
    image_name = image.split('/')[-1]
    if image_name + '.d.jpg' in file_names:
        print_info('{} 文件已存在!'.format(image_name))
        return
    img = cv2.imread(image)
    img_width = img.shape[0] / 120
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        i1_pt1 = (int(x_min), int(y_min))
        i1_pt2 = (int(x_max), int(y_max))
        cv2.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, thickness=img_width, color=(255, 0, 255))
    cv2.imwrite(os.path.join(out_folder, image_name + '.d.jpg'), img)  # 画图


if __name__ == '__main__':
    img_folder = os.path.join(IMG_DATA, 'logAll-0717')
    out_folder = os.path.join(IMG_DATA, 'logAll-0717-out')
    mkdir_if_not_exist(out_folder)
    process_anno_folder(img_folder, out_folder)
