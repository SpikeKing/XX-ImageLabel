#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/4
"""
import numpy as np
from mxnet.gluon.data import dataset
from mxnet.image import image

from contents.content_tags import CONTENT_TAGS
from utils.project_utils import *


class TripletDataset(dataset.Dataset):
    def __init__(self, data_folder=None, data_file=None, transform=None):
        self._flag = 1  # 彩色图片
        self.class_names = list(CONTENT_TAGS.keys())  # 全部类别, 27个类别

        self.data_file = data_file  # label file
        self.data_folder = data_folder  # img folder

        self.items = []
        self._transform = transform
        self._list_images()

    def __len__(self):
        return len(self.items)

    def get_n_class(self):
        return len(self.class_names)

    def _list_images(self):
        print('category num: {}'.format(len(self.class_names)))

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
                self.items.append((name_path_dict[name], oh_label))
        print('num of data: {}'.format(len(self.items)))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label


def main():
    # get_train_data()
    td = TripletDataset()
    print(td.get_n_class())
    for i in td:
        print(i[0])
        print(i[1])
        break


if __name__ == '__main__':
    main()
