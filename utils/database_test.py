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


def process_anno_folder(img_folder, out_folder):
    """
    处理标记文件夹
    :param img_folder:
    :param out_folder:
    :return:
    """
    img_dict = format_img_and_anno(img_folder)
    _, file_names = traverse_dir_files(out_folder)
    for img_name in img_dict:
        (img_p, anno_p) = img_dict[img_name]
        draw_img(img_p, read_anno_xml(anno_p), out_folder, file_names)


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
