#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""

from PIL import Image

from root_dir import IMG_DATA
from utils.dtc_utils import read_anno_xml, draw_boxes_simple
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
        # draw_img(img_p, read_anno_xml(anno_p), out_folder, file_names)


def format_img_and_anno(img_folder):
    """
    格式化输出。图片和标注文件夹
    :param img_folder: 图片文件夹
    :return:
    """
    file_paths, file_names = traverse_dir_files(img_folder)
    img_dict = dict()  # 将标注和图片路径，生成一个字典

    count = 0
    for file_path, file_name in zip(file_paths, file_names):
        if file_name.endswith('.jpg'):
            name = file_name.replace('.jpg', '')
            if name not in img_dict:
                img_dict[name] = (None, None)
            (img_p, anno_p) = img_dict[name]
            img_dict[name] = (file_path, anno_p)

        if file_name.endswith('.xml'):
            name = file_name.replace('.xml', '')
            if name not in img_dict:
                img_dict[name] = (None, None)
            (img_p, anno_p) = img_dict[name]
            img_dict[name] = (img_p, file_path)
            if not img_p:
                break

    print_info('图片数: {}'.format(len(img_dict.keys())))
    return img_dict


# def draw_img(image, boxes, out_folder, file_names):
#     image_name = image.split('/')[-1]
#     if image_name + '.d.jpg' in file_names:
#         print_info('{} 文件已存在!'.format(image_name))
#         return
#     img = cv2.imread(image)
#     img_width = img.shape[0] / 120
#     for box in boxes:
#         x_min, y_min, x_max, y_max = box
#         i1_pt1 = (int(x_min), int(y_min))
#         i1_pt2 = (int(x_max), int(y_max))
#         cv2.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, thickness=img_width, color=(255, 0, 255))
#     cv2.imwrite(os.path.join(out_folder, image_name + '.d.jpg'), img)  # 画图


if __name__ == '__main__':
    img_folder = os.path.join(IMG_DATA, 'jiaotong-0727')
    out_folder = os.path.join(IMG_DATA, 'jiaotong-0727-out')
    mkdir_if_not_exist(out_folder)
    process_anno_folder(img_folder, out_folder)
