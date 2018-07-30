#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""

import colorsys
import os

from PIL import ImageFont, ImageDraw, Image

from darknet import DarkNet
from face.yolov3_mxnet.dir_consts import CONFIGS_DATA, MODEL_DATA
from root_dir import IMG_DATA
from utils import *

image_name = 0


def generate_bboxes(img_data, bbox_raw, input_dim):
    """
    将原始数据转换为框
    :param img_data: 图片数据
    :param bbox_raw: 原始数据
    :param input_dim: 图片维度
    :return: 框, 置信度, 类别
    """
    img_size = nd.array([(img_data.shape[1], img_data.shape[0])])  # 图片高和宽
    img_size_all = nd.tile(img_size, 2)  # 重复两次, 生成4个坐标，对应(xmin, ymin, xmax, ymax)
    img_size_list = img_size_all[bbox_raw[:, 0], :]  # 重复生成多行，每个框对应1行

    sf = nd.min(input_dim / img_size_list, axis=1).reshape((-1, 1))  # 压缩比例
    sw_dim_list = sf * img_size_list[:, 0].reshape((-1, 1))  # 宽比例
    sh_dim_list = sf * img_size_list[:, 1].reshape((-1, 1))  # 高比例

    bbox_raw[:, [1, 3]] -= (input_dim - sw_dim_list) / 2  # 1,3宽
    bbox_raw[:, [2, 4]] -= (input_dim - sh_dim_list) / 2  # 2,4高
    bbox_raw[:, 1:5] /= sf  # 等比除法，还原

    for i in range(bbox_raw.shape[0]):
        bbox_raw[i, [1, 3]] = nd.clip(bbox_raw[i, [1, 3]], a_min=0.0, a_max=img_size_list[i][0].asscalar())
        bbox_raw[i, [2, 4]] = nd.clip(bbox_raw[i, [2, 4]], a_min=0.0, a_max=img_size_list[i][1].asscalar())

    bbox_raw = bbox_raw.asnumpy()

    boxes_, scores_, classes_ = [], [], []

    for bbox in bbox_raw:  # [图片索引, xmin, ymin, xmax, ymax, conf, prob, class_id]
        boxes_.append(bbox[1:5])
        scores_.append(bbox[5] * bbox[6])
        classes_.append(bbox[7])

    return boxes_, scores_, classes_


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


def main():
    img_path = os.path.join(IMG_DATA, 'hand_and_car.jpg')  # 图片路径
    params_path = os.path.join(MODEL_DATA, 'yolov3.weights')  # YOLO v3 权重文件
    classes_path = os.path.join(CONFIGS_DATA, 'coco.names')  # 类别文件

    batch_size = 16  # 批次数
    confidence = 0.15  # 置信度
    nms_thresh = 0.20  # NMS阈值
    input_dim = 416  # YOLOv3的检测尺寸

    classes_name = load_classes(classes_path)  # 加载类别目录
    num_classes = len(classes_name)  # 类别数

    gpu = '1'  # GPU
    gpu = [int(x) for x in gpu.replace(" ", "").split(",")]
    ctx = try_gpu(gpu)[0]  # 选择ctx

    net = DarkNet(input_dim=input_dim, num_classes=num_classes)  # 基础网络DarkNet
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)])  # anchors

    net.initialize(ctx=ctx)

    # 加载模型
    if params_path.endswith(".params"):
        net.load_params(params_path)
    elif params_path.endswith(".weights"):
        tmp_batch = nd.uniform(shape=(1, 3, input_dim, input_dim), ctx=ctx)
        net(tmp_batch)
        net.load_weights(params_path, fine_tune=False)
    else:
        print("params {} load error!".format(params_path))
        exit()
    print("load params: {}".format(params_path))
    net.hybridize()

    image_data = cv2.imread(img_path)  # 读取图片数据
    image_reform = prep_image(image_data, input_dim)
    image_arr = nd.array([image_reform], ctx=ctx)
    prediction = predict_transform(net(image_arr), input_dim, anchors)

    pred_res = write_results(prediction, num_classes, confidence=confidence, nms_conf=nms_thresh)

    hsv_tuples = [(float(x) / len(classes_name), 1., 1.)
                  for x in range(len(classes_name))]  # 不同颜色
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
    np.random.seed(10101)
    np.random.shuffle(colors)  # 随机颜色
    np.random.seed(None)
    img_data = Image.open(img_path)
    boxes, scores, classes = generate_bboxes(image_data, pred_res, input_dim=input_dim)
    image = draw_boxes(img_data, boxes, scores, classes, colors, classes_name)
    image.show()
    print(pred_res)


if __name__ == '__main__':
    main()
