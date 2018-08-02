#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/1
"""
import os
import sys
from PIL import Image

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from base.yolov3_mxnet.y3_class import Y3Model
from root_dir import IMG_DATA
from utils.dtc_utils import format_img_and_anno, read_anno_xml, map_classes
from utils.log_utils import print_info
from utils.project_utils import mkdir_if_not_exist, safe_div


def main():
    ym = Y3Model()
    img_folder = os.path.join(IMG_DATA, 'jiaotong-0727')
    right_folder = os.path.join(IMG_DATA, 'jiaotong-0727-right')
    wrong_folder = os.path.join(IMG_DATA, 'jiaotong-0727-wrong')
    mkdir_if_not_exist(right_folder)
    mkdir_if_not_exist(wrong_folder)
    img_dict = format_img_and_anno(img_folder)

    r_count = 0
    all_count = 0
    for count, img_name in enumerate(img_dict):
        (img_p, anno_p) = img_dict[img_name]
        # print(img_p)
        try:
            tag_res, img_box = ym.detect_img(img_p, True)
        except Exception as e:
            continue

        all_count += 1
        p_classes = set(tag_res.keys())

        _, t_classes = read_anno_xml(anno_p)
        merge_dict = {'truck': 'car', 'bus': 'car', 'car': 'car'}
        t_classes = map_classes(merge_dict, t_classes)  # 合并类别
        traffic_names = ['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat']
        t_classes = set(t_classes) & set(traffic_names)

        img_name = img_p.split('/')[-1]
        is_right = False
        if p_classes and p_classes.issubset(t_classes):
            r_count += 1
            img_box.save(os.path.join(right_folder, img_name + '.d.jpg'))
            is_right = True
        elif not p_classes and not t_classes:
            r_count += 1
            if not img_box:
                img_box = Image.open(img_p)
            img_box.save(os.path.join(right_folder, img_name + '.d.jpg'))
            is_right = True
        else:
            if not img_box:
                img_box = Image.open(img_p)
            img_box.save(os.path.join(wrong_folder, img_name + '.d.jpg'))

        print_info('P: {}, T: {}, {}'.format(list(p_classes), list(t_classes), '正确' if is_right else '错误'))
        # if count == 10:
        #     break

    right_ratio = safe_div(r_count, all_count)
    print_info('正确: {}, 全部: {}, 准确率: {}'.format(r_count, all_count, right_ratio))


if __name__ == '__main__':
    main()
