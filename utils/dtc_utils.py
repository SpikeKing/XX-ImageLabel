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
from PIL import ImageFont, ImageDraw, Image

from root_dir import ROOT_DIR, FONT_DATA
from utils.log_utils import print_info
from utils.project_utils import read_file, traverse_dir_files


def make_line_colors(n_color=20, bias=1.0, alpha=1.0):
    """
    创建检测框的颜色集合，通过HSV创建区分度更好的颜色

    :param n_color: 颜色数量
    :param alpha: 透明度
    :param bias: 颜色的偏移
    :return: 颜色集合
    """
    alpha = int(alpha * 256)
    hsv_tuples = [(float(x) / n_color, 1., 1.) for x in range(n_color)]  # 不同颜色
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    mul_v = 255 * bias
    colors = list(map(lambda x: (int(x[0] * mul_v), int(x[1] * mul_v), int(x[2] * mul_v), alpha), colors))  # RGB
    np.random.seed(10101)
    np.random.shuffle(colors)  # 随机颜色
    np.random.seed(None)
    return colors


def filter_sbox(image_size, box_group, ratio=0.003):
    """
    过滤检测图像中，比较小的框
    :param image_size: 图片宽高, 如(416, 416)
    :param box_group: 框组合, 如(boxes, scores, classes)
    :param ratio: 阈值
    :return: 过滤后的框组合
    """
    # 过滤小于0.004的图像
    boxes, scores, classes = box_group
    t_boxes, t_scores, t_classes = [], [], []
    for out_box, out_score, out_class in zip(boxes, scores, classes):
        img_size = image_size[1] * image_size[0]
        # print_info('图片大小: %s' % img_size)
        top, left, bottom, right = out_box
        top, left, bottom, right = int(top), int(left), int(bottom), int(right)
        box_ratio = abs(top - bottom) * abs(left - right) / float(img_size)
        if box_ratio > ratio:
            t_boxes.append(out_box)
            t_scores.append(out_score)
            t_classes.append(out_class)
    return t_boxes, t_scores, t_classes


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


def draw_boxes_simple(image, boxes, scores, class_names, colors_dict, is_alpha=False):
    """
    在PIL.Image图像中，绘制预测框和标签

    :param image: 图片
    :param boxes: 框数列表，ymin, xmin, ymax, xmax
    :param scores: 置信度列表
    :param class_names: 类别名称
    :param colors_dict: 颜色集合
    :param is_alpha: 是否透明
    :return: 绘制之后的Image图像
    """
    # print('框数: {}'.format(len(boxes)))  # 画框的数量

    if is_alpha:
        image = image.convert("RGBA")  # 转换为透明模式
        image_ = image.copy()  # 复制图片，用于处理
    else:
        image_ = image

    font = ImageFont.truetype(font=os.path.join(FONT_DATA, 'FiraMono-Medium.otf'),
                              size=np.floor(2e-2 * image_.size[1] + 0.5).astype('int32'))  # 字体
    thickness = (image_.size[0] + image_.size[1]) // 512  # 边框的大小

    for j, p_class in reversed(list(enumerate(class_names))):
        score = scores[j]  # 置信度
        box = boxes[j]  # 框

        try:
            label = '{} {:.2f}'.format(p_class, score)  # 标签
        except Exception as e:
            label = '{} {}'.format(p_class, score)  # 标签

        draw = ImageDraw.Draw(image_)  # 画图
        label_size = draw.textsize(label, font)  # 标签文字

        left, top, right, bottom = box  # Box中的信息: xmin, ymin, xmax, ymax
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_.size[0], np.floor(right + 0.5).astype('int32'))

        # print_info('框{}: {}, {}'.format(j, (p_class, score), (left, top, right, bottom)))  # 边框

        if top - label_size[1] >= 0:  # 标签文字
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for j in range(thickness):  # 一笔一笔的画框
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors_dict[p_class])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors_dict[p_class])  # 标签背景
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 标签字体
        del draw

    if is_alpha:
        image_ = Image.alpha_composite(image, image_)  # 添加透明效果，透明效果在颜色中设置

    return image_


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
    # print('框数: {}'.format(len(boxes)))  # 画框的数量

    font = ImageFont.truetype(font=os.path.join(FONT_DATA, 'FiraMono-Medium.otf'),
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
    thickness = (image.size[0] + image.size[1]) // 512  # 边框的大小

    for j, c in reversed(list(enumerate(classes))):
        color_i = int(c % len(colors))
        p_class = class_names[int(c)]  # 预测类别
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
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[color_i])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[color_i])  # 标签背景
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 标签字体
        del draw

    return image


if __name__ == '__main__':
    fp = os.path.join(ROOT_DIR, "img_data/logAll-0717/ErTong_6.xml")
    boxes_list, _ = read_anno_xml(fp)
    print(boxes_list)
