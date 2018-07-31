#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20

检测的公共函数文件
"""
import colorsys
import os
import numpy as np

import xmltodict
from PIL import ImageFont, ImageDraw

from img_downloader.xx_downloader import print_info, traverse_dir_files
from root_dir import ROOT_DIR, FONT_DATA
from utils.project_utils import read_file


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

    print_info('图片数: {}'.format(len(img_dict.keys())))
    return img_dict


def read_anno_xml(xml_file):
    """
    读取XML的文件
    :param xml_file: XML的文件
    :return:
    """
    xml_dict = xmltodict.parse(read_file(xml_file, mode='one'))
    name_list = []
    boxes_list = []

    try:
        annotation = xml_dict['annotation']
        if 'object' in annotation:
            objects = annotation['object']
            if isinstance(objects, list):
                for object in objects:
                    name = object['name']
                    bndbox = object['bndbox']
                    name_list.append(name)
                    boxes_list.append([bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']])
            else:
                name = objects['name']
                bndbox = objects['bndbox']
                name_list.append(name)
                boxes_list.append([bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']])
    except Exception as e:
        pass

    return boxes_list, name_list


def draw_boxes_simple(image, boxes, scores, class_names):
    """
    在PIL.Image图像中，绘制预测框和标签

    :param image: 图片
    :param boxes: 框数列表，ymin, xmin, ymax, xmax
    :param scores: 置信度列表
    :param class_names: 类别名称
    :return: 绘制之后的Image图像
    """
    print('框数: {}'.format(len(boxes)))  # 画框的数量

    font = ImageFont.truetype(font=os.path.join(FONT_DATA, 'FiraMono-Medium.otf'),
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
    thickness = (image.size[0] + image.size[1]) // 512  # 边框的大小

    n_color = 20
    hsv_tuples = [(float(x) / n_color, 1., 1.) for x in range(n_color)]  # 不同颜色
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
    np.random.seed(10101)
    np.random.shuffle(colors)  # 随机颜色
    np.random.seed(None)

    uni_cns = list(set(class_names))  # 去重, 用于确定颜色

    for j, p_class in reversed(list(enumerate(class_names))):
        c = uni_cns.index(p_class) % len(colors)

        score = scores[j]  # 置信度
        box = boxes[j]  # 框

        try:
            label = '{} {:.2f}'.format(p_class, score)  # 标签
        except:
            print_info('类别异常')
            label = '{} {:.2f}'.format('异常', score)  # 标签
        draw = ImageDraw.Draw(image)  # 画图
        label_size = draw.textsize(label, font)  # 标签文字

        left, top, right, bottom = box  # Box中的信息: xmin, ymin, xmax, ymax
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # print_info('框{}: {}, {}'.format(j, (p_class, score), (left, top, right, bottom)))  # 边框

        if top - label_size[1] >= 0:  # 标签文字
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for j in range(thickness):  # 一笔一笔的画框
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])  # 标签背景
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 标签字体
        del draw

    return image


def draw_boxes(image, boxes, scores, classes, colors, class_names):
    """
    在PIL.Image图像中，绘制预测框和标签

    :param image: 图片
    :param boxes: 框数列表，ymin, xmin, ymax, xmax
    :param scores: 置信度列表
    :param classes: 类别列表
    :param colors: 重新设置颜色
    :param class_names: 类别名称
    :return: 绘制之后的Image图像
    """
    print('框数: {}'.format(len(boxes)))  # 画框的数量

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
    thickness = (image.size[0] + image.size[1]) // 512  # 边框的大小

    for j, c in reversed(list(enumerate(classes))):
        c = int(c)
        p_class = class_names[c]  # 预测类别
        score = scores[j]  # 置信度
        box = boxes[j]  # 框

        label = '{} {:.2f}'.format(p_class, score)  # 标签
        draw = ImageDraw.Draw(image)  # 画图
        label_size = draw.textsize(label, font)  # 标签文字

        left, top, right, bottom = box  # Box中的信息: xmin, ymin, xmax, ymax

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # print_info('框{}: {}, {}'.format(j, (p_class, score), (left, top, right, bottom)))  # 边框

        if top - label_size[1] >= 0:  # 标签文字
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for j in range(thickness):  # 一笔一笔的画框
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])  # 标签背景
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 标签字体
        del draw

    return image


if __name__ == '__main__':
    fp = os.path.join(ROOT_DIR, "img_data/logAll-0717/ErTong_6.xml")
    boxes_list, _ = read_anno_xml(fp)
    print(boxes_list)
