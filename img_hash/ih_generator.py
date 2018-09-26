#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/19
"""
import os
import sys
import numpy as np

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from img_hash.dir_const import DATA_DIR
from img_hash.xu_pkg.im_hash_api import ImageHash
from root_dir import ROOT_DIR
from utils.project_utils import *


def to_binary(bit_list):
    # out = long(0)  # 必须指定为long，否则存储过少
    out = np.longlong(0)  # 必须指定为long，否则存储过少
    for bit in bit_list:
        out = (out << 1) | bit
    return out


def process_data():
    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    train_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    train_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别

    name_labels_dict = dict()
    data_lines = read_file_utf8(train_file)
    for data_line in data_lines:
        items = data_line.split('---')
        name = items[0]
        labels = items[1]
        name_labels_dict[name] = labels

    pathes, names = traverse_dir_files(train_folder)

    ih = ImageHash()
    name_list = []
    label_list = []
    bin_list = []
    data_list = []
    print('数据量: {}'.format(len(names)))
    for count, (path, name) in enumerate(zip(pathes, names)):
        res = ih.predict(path)
        cls_id, res_data = res  # 类别和数据
        oz_arr = np.where(res_data >= 0.5, 1.0, 0.0).astype(int)
        label = name_labels_dict[name]
        # oz_bin = np.apply_along_axis(to_binary, axis=0, arr=oz_arr)
        data_list.append(res_data)
        bin_list.append(oz_arr)
        name_list.append(name)
        label_list.append(label)
        if count % 1000 == 0:
            print(count)

    out_path = os.path.join(DATA_DIR, 'train.bin.npz')
    np.savez(out_path, b_list=bin_list, n_list=name_list,
             l_list=label_list, d_list=data_list)


def read_data():
    out_path = os.path.join(DATA_DIR, 'train.bin.npz')
    bin_data = np.load(out_path)
    bin_list = bin_data['b_list']
    name_list = bin_data['n_list']
    data_list = bin_data['d_list']
    label_list = bin_data['l_list']
    print(bin_list)
    print(name_list)
    print(data_list)
    print(label_list)


def main():
    process_data()
    # read_data()


if __name__ == '__main__':
    main()
