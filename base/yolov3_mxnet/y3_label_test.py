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
    none_folder = os.path.join(IMG_DATA, 'jiaotong-0727-none')
    mkdir_if_not_exist(right_folder, is_delete=True)
    mkdir_if_not_exist(wrong_folder, is_delete=True)
    mkdir_if_not_exist(none_folder, is_delete=True)
    img_dict = format_img_and_anno(img_folder)

    r_count = 0
    all_count = 0
    no_recall_count = 0

    for count, img_name in enumerate(img_dict):
        (img_p, anno_p) = img_dict[img_name]
        # print(img_p)
        try:
            tag_res, img_box = ym.detect_img(img_p, True)
        except Exception as e:
            continue

        w_tags = []
        for tag in tag_res.keys():
            if tag_res[tag] <= 0.01:
                print_info('删除Tag {} {}'.format(tag, tag_res[tag]))
                w_tags.append(tag)
        for tag in w_tags:
            tag_res.pop(tag, None)  # 小于1%的类别

        all_count += 1
        p_classes = set(tag_res.keys())

        _, t_classes = read_anno_xml(anno_p)
        merge_dict = {'truck': 'car', 'bus': 'car', 'car': 'car', 'motorbike': 'bicycle'}  # 合并类别
        t_classes = map_classes(merge_dict, t_classes)  # 合并类别
        traffic_names = ['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat']
        t_classes = set(t_classes) & set(traffic_names)

        img_name = img_p.split('/')[-1]
        is_right = False

        if p_classes and p_classes.issubset(t_classes):  # 检测正确
            r_count += 1
            img_box.save(os.path.join(right_folder, img_name + '.d.jpg'))
            is_right = True
        elif not p_classes and not t_classes:  # 空，检测正确
            r_count += 1
            if not img_box:
                img_box = Image.open(img_p)
            img_box.save(os.path.join(right_folder, img_name + '.d.jpg'))
            is_right = True
        elif not p_classes and t_classes:  # 检测为空，实际有类
            if not img_box:
                img_box = Image.open(img_p)
            img_box.save(os.path.join(none_folder, img_name + '.d.jpg'))
            no_recall_count += 1  # 未召回
            is_right = True
        else:  # 其他，检测错误
            if not img_box:
                img_box = Image.open(img_p)
            img_box.save(os.path.join(wrong_folder, img_name + '.d.jpg'))

        print_info('P: {}, T: {}, {}'.format(list(p_classes), list(t_classes), '正确' if is_right else '错误'))

    right_ratio = safe_div(r_count, all_count)
    print_info('正确: {}, 全部: {}, 未召回: {}, 准确率: {}'.format(r_count, all_count, no_recall_count, right_ratio))


if __name__ == '__main__':
    main()
