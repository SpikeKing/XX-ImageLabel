#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import colorsys
import os
import numpy as np

from PIL import ImageFont, ImageDraw
from keras import Input
from keras import backend as K
from timeit import default_timer as timer

from face.yolov3.core.model import yolo_body, yolo_eval
from face.yolov3.core.utils import letterbox_image

from utils.log_utils import print_info
from utils.project_utils import write_line


class YoloV3(object):
    def __init__(self, model_path='model_data/ep074-loss26.535-val_loss27.370.h5',
                 classes_path='configs/wider_classes.txt',
                 anchors_path='configs/yolo_anchors.txt'):

        self.model_path = model_path  # 模型文件
        self.classes_path = classes_path  # 类别文件
        self.anchors_path = anchors_path  # anchors文件
        print_info('模型: {}'.format(model_path))

        self.score = 0.25
        self.iou = 0.30
        print_info('置信度: {}, IoU: {}'.format(self.score, self.iou))  # 输出参数

        self.class_names = self._get_class()  # 获取类别
        self.anchors = self._get_anchors()  # 获取anchor
        self.sess = K.get_session()
        print_info('anchors: {}, 类别: {}'.format(len(self.anchors), len(self.class_names)))

        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        print_info('图片尺寸: {}'.format(self.model_image_size))  # 输出参数

        self.yolo_model = None
        self.colors = None
        self.input_image_shape = None

        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 转换~
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        num_classes = len(self.class_names)  # 类别数

        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), 3, num_classes)  # Yolo结构
        self.yolo_model.load_weights(model_path)  # 参数

        hsv_tuples = [(float(x) / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]  # 不同颜色
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(self.colors)  # 随机颜色
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))  # 根据检测参数，过滤框
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image_facets(self, image):
        """
        检测图片的数据细节，检测框、置信度、类别
        :param image: PIL图片
        :return: 检测框、置信度、类别
        """
        start_t = timer()  # 起始时间
        print_info('{}检测开始{}'.format('-' * 10, '-' * 10))

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)  # 强制转换为32的倍数

        image_data = np.array(boxed_image, dtype='float32')  # 图片数据
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,  # 输入检测图片
                self.input_image_shape: [image.size[1], image.size[0]],  # 输入检测尺寸
                K.learning_phase(): 0  # 学习率, 0表示测试, 1表示训练
            })

        print_info('检测时间: {:.4f} 秒'.format(timer() - start_t))  # 检测执行时间
        print_info('检测物体数: {} 个'.format(len(out_boxes)))  # 检测结果
        print_info('{}检测结束{}'.format('-' * 10, '-' * 10))

        return out_boxes, out_scores, out_classes

    def draw_boxes(self, image, boxes, scores, classes, colors=None):
        """
        在PIL.Image图像中，绘制预测框和标签

        :param image: 图片
        :param boxes: 框数列表，ymin, xmin, ymax, xmax
        :param scores: 置信度列表
        :param classes: 类别列表
        :param colors: 重新设置颜色
        :return: 绘制之后的Image图像
        """
        print_info('框数: {}'.format(len(boxes)))  # 画框的数量

        if colors:
            self.colors = colors

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 边框的大小

        for j, c in reversed(list(enumerate(classes))):
            p_class = self.class_names[c]  # 预测类别
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

            print_info('框{}: {}, {}'.format(j, (p_class, score), (left, top, right, bottom)))  # 边框

            if top - label_size[1] >= 0:  # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):  # 一笔一笔的画框
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])  # 标签背景
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 标签字体
            del draw

        return image

    def detect_image(self, image, image_name=None, out_file=None):
        start = timer()  # 起始时间

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        # print('detector size {}'.format(image_data.shape))
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,  # 输入检测图片
                self.input_image_shape: [image.size[1], image.size[0]],  # 输入检测尺寸
                K.learning_phase(): 0  # 学习率, 0表示测试, 1表示训练
            })

        if image_name and out_file:
            boxes_list = [image_name]
            for box, clazz in zip(out_boxes, out_classes):
                y_min, x_min, y_max, x_max = box.tolist()
                line_list = [x_min, y_min, x_max, y_max]
                line_list = [str(int(x)) for x in line_list]
                line_list.append(str(clazz))
                boxes_list.append(','.join(line_list))
            write_line(out_file, ' '.join(boxes_list))

        # 过滤小于0.004的图像
        # tmp_boxes, tmp_scores, tmp_classes = [], [], []
        # for out_box, out_score, out_class in zip(out_boxes, out_scores, out_classes):
        #     img_size = image.size[1] * image.size[0]
        #     # print_info('图片大小: %s' % img_size)
        #     top, left, bottom, right = out_box
        #     box_ratio = abs(top - bottom) * abs(left - right) / img_size
        #     if box_ratio > 0.004:
        #         tmp_boxes.append(out_box)
        #         tmp_scores.append(out_score)
        #         tmp_classes.append(out_class)
        # out_boxes, out_scores, out_classes = tmp_boxes, tmp_scores, tmp_classes

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 检测出的框

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 厚度

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  # 类别
            box = out_boxes[i]  # 框
            score = out_scores[i]  # 执行度

            label = '{} {:.2f}'.format(predicted_class, score)  # 标签
            draw = ImageDraw.Draw(image)  # 画图
            label_size = draw.textsize(label, font)  # 标签文字

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))  # 边框

            if top - label_size[1] >= 0:  # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):  # 画框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(  # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
            del draw

        end = timer()
        print(end - start)  # 检测执行时间
        return image

    def close_session(self):
        self.sess.close()
