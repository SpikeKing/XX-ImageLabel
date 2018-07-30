#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""
from face.yolov3.module_dir import CONFIGS_DATA
from face.yolov3.yolov3_dir import OUTPUT_DATA, MODEL_DATA
from root_dir import IMG_DATA, ROOT_DIR
from utils.log_utils import print_info
from utils.project_utils import mkdir_if_not_exist, traverse_dir_files, write_line

"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from core.model import yolo_eval, yolo_body
from core.utils import letterbox_image


class Yolov3Predictor(object):
    def __init__(self, model_path, classes_path, anchors_path):

        self.model_path = model_path  # 模型文件
        self.classes_path = classes_path  # 类别文件

        self.anchors_path = anchors_path  # Anchors
        self.score = 0.25
        self.iou = 0.20
        self.class_names = self._get_class()  # 获取类别
        self.anchors = self._get_anchors()  # 获取anchor
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
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

        num_anchors = len(self.anchors)  # anchors的数量
        num_classes = len(self.class_names)  # 类别数

        # 加载模型参数
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), 3, num_classes)
        self.yolo_model.load_weights(model_path)  # 核心模型

        print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))

        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]  # 不同颜色
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # 根据检测参数，过滤框
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names),
                                           self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

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
                line_list = box.tolist()
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
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
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
        # print(end - start)  # 检测执行时间
        return image

    def close_session(self):
        self.sess.close()


def detect_img_folder(img_folder, out_folder, yolo):
    mkdir_if_not_exist(out_folder)
    path_list, name_list = traverse_dir_files(img_folder)
    print_info('图片数: %s' % len(path_list))

    _, imgs_names = traverse_dir_files(out_folder)

    count = 0
    for path, name in zip(path_list, name_list):
        if path.endswith('.gif'):
            continue

        out_name = name + '.d.jpg'
        if out_name in imgs_names:
            print_info('已检测: %s' % name)
            continue

        print_info('检测图片: %s' % name)

        try:
            image = Image.open(path)
            out_file = os.path.join(ROOT_DIR, 'face', 'yolov3', 'output_data', 'logAll_res.txt')
            r_image = yolo.detect_image(image, ('logAll/' + name), out_file)
            r_image.save(os.path.join(out_folder, name + '.d.jpg'))
        except Exception as e:
            print(e)
            pass

        count += 1
        if count % 100 == 0:
            print_info('已检测: %s' % count)
    yolo.close_session()


def detect_img_for_test(yolo):
    file_name = 'hand_and_car.jpg'
    img_path = os.path.join(IMG_DATA, 'hand_and_car.jpg')
    image = Image.open(img_path)
    out_file = os.path.join(ROOT_DIR, 'face', 'yolov3', 'output_data', 'logTest_res.txt')
    # r_image = yolo.detect_image(image, ('logAll/' + file_name), out_file)
    r_image = yolo.detect_image(image)
    r_image.show()
    # output_folder = os.path.join(OUTPUT_DATA, 'logStars')
    # mkdir_if_not_exist(output_folder)
    # r_image.save(os.path.join(output_folder, file_name + '.d.jpg'))
    yolo.close_session()


if __name__ == '__main__':
    # model_path = os.path.join(MODEL_DATA, 'ep030-loss22.691-val_loss24.034.h5')
    # classes_path = 'configs/wider_classes.txt',
    # anchors_path = 'configs/yolo_anchors.txt'

    model_path = os.path.join(MODEL_DATA, 'yolo_weights.h5')
    classes_path = os.path.join(CONFIGS_DATA, 'coco_classes.txt')
    anchors_path = os.path.join(CONFIGS_DATA, 'yolo_anchors.txt')

    yolo = Yolov3Predictor(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path)
    img_folder = os.path.join(IMG_DATA, 'logAll')
    out_folder = os.path.join(OUTPUT_DATA, 'logAll')
    # detect_img_folder(img_folder, out_folder, yolo)
    detect_img_for_test(yolo)
