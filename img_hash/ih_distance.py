#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/26
"""
import os
import sys
import numpy as np
from PIL import Image

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from img_hash.dir_const import DATA_DIR
from root_dir import ROOT_DIR
from utils.project_utils import *


def load_data():
    out_path = os.path.join(DATA_DIR, 'tr_train.bin.npz')
    bin_data = np.load(out_path)
    bin_list = bin_data['b_list']
    name_list = bin_data['n_list']
    data_list = bin_data['d_list']
    label_list = bin_data['l_list']
    return bin_list, name_list, data_list, label_list


def distance_ab(a, b):
    res = np.count_nonzero(a - b)
    return res


def distance(out_folder, data_set, bin_data, name):
    bin_list, name_list, data_list, label_list = data_set

    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    train_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    pathes, names = traverse_dir_files(train_folder)

    name_path_dict = dict()
    for path, name in zip(pathes, names):
        name_path_dict[name] = path

    count_dist_dict = dict()
    for count, bl in enumerate(bin_list):
        count_dist_dict[count] = distance_ab(bin_data, bl)

    count_dist_list = sort_dict_by_value(count_dist_dict, reverse=False)

    path_list = []
    image_ths = 12
    img_w, img_h = 4, 3
    for i, (count, dist) in enumerate(count_dist_list):
        name = name_list[count]
        path_list.append(name_path_dict[name])
        if i == image_ths:
            break

    images = [Image.open(p) for p in path_list]

    new_im = Image.new('RGB', (416 * img_w, 416 * img_h), color=(255, 255, 255))

    x_offset, y_offset = 0, 0
    for i in range(img_h):
        for j in range(img_w):
            im = images[i * img_w + j]
            new_im.paste(im, (x_offset, y_offset))
            x_offset += 416
        y_offset += 416
        x_offset = 0

    new_im.save(os.path.join(out_folder, name + '.res.jpg'))


def main():
    out_folder = os.path.join(DATA_DIR, 'im_hash_res')
    mkdir_if_not_exist(out_folder, is_delete=True)
    data_set = load_data()
    bin_list, name_list, data_list, label_list = data_set
    for count, (bin_data, name_data) in enumerate(zip(bin_list, name_list)):
        distance(out_folder, data_set, bin_data, name_data)
        if count % 100 == 0:
            print(count)


if __name__ == '__main__':
    main()
