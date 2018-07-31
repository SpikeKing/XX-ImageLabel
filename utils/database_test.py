#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""

from PIL import Image

from root_dir import IMG_DATA
from utils.dtc_utils import read_anno_xml, draw_boxes_simple, format_img_and_anno
from utils.log_utils import print_info
from utils.project_utils import *


def process_anno_folder(img_folder, out_folder):
    img_dict = format_img_and_anno(img_folder)
    _, file_names = traverse_dir_files(out_folder)

    for count, img_name in enumerate(img_dict.keys()):
        print_info('-' * 50)

        (img_p, anno_p) = img_dict[img_name]
        if not img_p or not anno_p:
            print_info('图片: {} 异常'.format(img_name))
            continue
        if img_name in file_names:
            print_info('图片: {} 已存在'.format(img_name))
        else:
            print_info('图片: {}'.format(img_name))

        image_data = Image.open(img_p)
        boxes_list, name_list = read_anno_xml(anno_p)
        prob_list = [1.0 for x in range(len(boxes_list))]
        img_boxes = draw_boxes_simple(image_data, boxes_list, prob_list, name_list)

        img_boxes.save(os.path.join(out_folder, img_name + '.d.jpg'))


if __name__ == '__main__':
    img_folder = os.path.join(IMG_DATA, 'jiaotong-0727')
    out_folder = os.path.join(IMG_DATA, 'jiaotong-0727-out')
    mkdir_if_not_exist(out_folder)
    process_anno_folder(img_folder, out_folder)
