#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/4
"""

import mxnet as mx
import numpy as np

from mxnet.gluon.data import dataset
from mxnet.image import image

from contents.content_tags import CONTENT_TAGS
from utils.project_utils import *


class TripletDataset(dataset.Dataset):
    """
    MultiLabel的TripletLoss数据集
    生成anchor、positive、negative示例
    """

    def __init__(self, data_folder=None, data_file=None, transform=None, saved_path=None):
        """
        构造器。
        :param data_folder: 数据文件夹
        :param data_file: 数据文件
        :param transform: 数据扩充
        :param saved_path: 已计算的TripletLoss对，加快速度
        """
        self._flag = 1  # 彩色图片
        self.class_names = list(CONTENT_TAGS.keys())  # 全部类别, 27个类别
        self._print_info(self.class_names)

        if data_file:  # 加载数据文件
            self.data_file = data_file  # 图片和标签的文本
            self.np_saved = os.path.join(data_file + ".tp.npz")

        self.data_folder = data_folder  # 图片数据集

        self.items = []  # 项目列表
        self.tp_list = []  # TP列表
        self._transform = transform  # 转换器，用于Data Augmentation

        self._list_images()  # 生成图片列表

        if not saved_path:
            self._create_pairs()  # 生成TP的数据对
        else:
            self.np_saved = os.path.join(data_file + ".tp.npz")
            self._load_pairs()  # 加载pair组

    @staticmethod
    def _print_info(s):
        """
        打印INFO信息
        """
        print('[Info] {}'.format(s))

    def __len__(self):
        """
        图片个数
        """
        num = len(self.tp_list[0])
        # self._print_info('样本数: {}'.format(num))
        return num

    def get_class_names(self):
        """
        类别列表
        """
        return self.class_names

    def get_n_class(self):
        """
        类别数
        """
        return len(self.class_names)

    def _create_pairs(self):
        """
        创建anchor，positive，negative示例对；
        MultiLabel的label全部相等，才能匹配。
        :return: Triplet Loss示例对
        """
        label_index_dict = dict()  # 不同index的索引值

        # 第1次循环，创建各个标签的索引值
        for i, item in enumerate(self.items):
            img, label = item
            indexes = np.where(label == 1)[0]
            for idx in indexes.tolist():
                if idx not in label_index_dict:
                    label_index_dict[idx] = []
                label_index_dict[idx].append(i)

        data_list, label_list = [], []
        self._print_info('创建三元组列表开始...')

        start_time = time.time()
        for i, item in enumerate(self.items):
            img, label = item
            indexes = np.where(label == 1)[0].tolist()

            p_labels = set()
            rn_labels = set()
            for idx in indexes:
                if not p_labels:
                    p_labels = set(label_index_dict[idx])
                else:
                    p_labels &= set(label_index_dict[idx])  # positive用交集

                if not rn_labels:
                    rn_labels = set(label_index_dict[idx])
                else:
                    rn_labels |= set(label_index_dict[idx])  # negative用并集

            p_labels = list(set(p_labels))
            a = set(np.arange(len(self.items)))
            b = set(rn_labels)
            n_labels = list(a - b)  # 排除全部并集
            p_index = random.choice(p_labels)  # 正示例索引
            n_index = random.choice(n_labels)  # 负示例索引

            data_pair = [i, p_index, n_index]
            label_pair = np.asarray([label, self.items[p_index][1], self.items[n_index][1]])

            data_list.append(data_pair)
            label_list.append(label_pair)

            if (i + 1) % 100 == 0:
                self._print_info('tl count: {}'.format(i))
                # break

        elapsed_time = time.time() - start_time
        self._print_info('耗时: {:.2f} 秒'.format(elapsed_time))
        self.tp_list = [data_list, label_list]  # 由data和label的list组成

        np.savez(self.np_saved, data=data_list, labels=label_list)  # 数据执行较慢，存储起来，复用

    def _list_images(self):
        """
        列出图片数据
        """
        # self._print_info('类别数: {}'.format(len(self.class_names)))

        pathes, names = traverse_dir_files(self.data_folder)
        name_path_dict = dict(zip(names, pathes))

        ex_tags = [u'人像', u'女', u'男', u'男女', u'游乐园']  # 去除无效标签
        map_tags = {u'抗老': u'护肤', u'美妆教程': u'美妆', u'旅行': u'旅游',
                    u'跑步': u'健身', u'洁面（卸妆）': u'护肤', u'预告资讯': u'影视综艺',
                    u'手霜': u'护肤', u'爽身粉（水）': u'美体', u'当季流行': u'穿搭',
                    u'app': u'数码', u'臀部': u'健身'}  # 映射为一级标签

        data_lines = read_file_utf8(self.data_file)
        for data_line in data_lines:
            name, tags_str, _ = data_line.split('---')
            tags = tags_str.split(',')
            oh_label = np.zeros((len(self.class_names)))
            for tag in tags:
                if tag in ex_tags:
                    continue
                if tag in map_tags.keys():
                    tag = map_tags[tag]
                index = self.class_names.index(tag)
                oh_label[index] = 1
            if name in name_path_dict.keys():
                self.items.append((name_path_dict[name], oh_label))

        # self._print_info('样本数: {}'.format(len(self.items)))

    def _load_pairs(self):
        """
        加载数据
        """
        npz_file = np.load(self.np_saved)
        self.tp_list = [npz_file['data'], npz_file['labels']]

    def __getitem__(self, idx):
        """
        迭代器的核心接口，返回数据
        """
        tp_list, l_list = self.tp_list
        tp_pair = tp_list[idx]
        l_pair = l_list[idx]
        imgs, labels = [], []

        for idx, img_idx in enumerate(tp_pair):
            img = image.imread(self.items[img_idx][0], self._flag)
            label = l_pair[idx]
            if self._transform is not None:
                img, label = self._transform(img), label
            img = mx.nd.expand_dims(img, axis=0)  # 增加1维，设置图片
            imgs.append(img)
            labels.append(label)

        data = mx.nd.concatenate(imgs, axis=0)  # 增加1维，设置图片
        labels = mx.nd.array(labels)

        return data, labels


class MultilabelDataset(dataset.Dataset):
    """
    Multilabel的数据集
    """

    def __init__(self, data_folder=None, data_file=None, transform=None):
        """
        构造器
        :param data_folder: 数据文件夹
        :param data_file: 数据文件
        :param transform: 转换器
        """
        self._flag = 1  # 彩色图片
        self.class_names = list(CONTENT_TAGS.keys())  # 全部类别, 27个类别

        self.data_file = data_file  # 数据文件
        self.data_folder = data_folder  # 图片文件夹

        self.items = []
        self._transform = transform
        self._list_images()

    @staticmethod
    def _print_info(s):
        """
        打印INFO信息
        """
        print('[Info] {}'.format(s))

    def __len__(self):
        """
        样本数
        """
        return len(self.items)

    def get_n_class(self):
        """
        类别数
        """
        return len(self.class_names)

    def _list_images(self):
        """
        转换为多标签的样本
        """
        self._print_info('类别数: {}'.format(len(self.class_names)))

        pathes, names = traverse_dir_files(self.data_folder)
        name_path_dict = dict(zip(names, pathes))

        ex_tags = [u'人像', u'女', u'男', u'男女', u'游乐园', ]
        map_tags = {u'抗老': u'护肤', u'美妆教程': u'美妆', u'旅行': u'旅游',
                    u'跑步': u'健身', u'洁面（卸妆）': u'护肤', u'预告资讯': u'影视综艺',
                    u'手霜': u'护肤', u'爽身粉（水）': u'美体', u'当季流行': u'穿搭',
                    u'app': u'数码', u'臀部': u'健身'}

        data_lines = read_file_utf8(self.data_file)
        for data_line in data_lines:
            name, tags_str, _ = data_line.split('---')
            tags = tags_str.split(',')
            oh_label = np.zeros((len(self.class_names)))
            for tag in tags:
                if tag in ex_tags:
                    continue
                if tag in map_tags.keys():
                    tag = map_tags[tag]

                index = self.class_names.index(tag)
                oh_label[index] = 1
            if name in name_path_dict.keys():
                self.items.append((name_path_dict[name], oh_label))  # item包含图片和标签
        self._print_info('样本数: {}'.format(len(self.items)))

    def __getitem__(self, idx):
        """
        迭代器
        """
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            img, label = self._transform(img), label
        return img, label
