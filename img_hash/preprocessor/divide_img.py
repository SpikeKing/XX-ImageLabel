#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/3
"""
import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from img_hash.dir_const import IMGS_DATA, DATA_DIR
from utils.project_utils import *


def read_tag_imgs(file_path):
    data_lines = read_file_utf8(file_path)
    res_dict = {}
    for data_line in data_lines:
        name, _, _ = data_line.split('---')
        res_dict[name] = data_line
    return res_dict


def main():
    pathes, names = traverse_dir_files(IMGS_DATA)

    train_path = os.path.join(DATA_DIR, 'img_tags_train.txt')
    val_path = os.path.join(DATA_DIR, 'img_tags_val.txt')
    test_path = os.path.join(DATA_DIR, 'img_tags_test.txt')
    train_dict = read_tag_imgs(train_path)
    val_dict = read_tag_imgs(val_path)
    test_dict = read_tag_imgs(test_path)

    train_dir = os.path.join(DATA_DIR, 'train_data')
    val_dir = os.path.join(DATA_DIR, 'val_data')
    test_dir = os.path.join(DATA_DIR, 'test_data')
    create_folder(train_dir)
    create_folder(val_dir)
    create_folder(test_dir)

    t_train_path = os.path.join(DATA_DIR, 't_img_tags_train.txt')
    t_val_path = os.path.join(DATA_DIR, 't_img_tags_val.txt')
    t_test_path = os.path.join(DATA_DIR, 't_img_tags_test.txt')
    create_file(t_train_path)
    create_file(t_val_path)
    create_file(t_test_path)

    for path, name in zip(pathes, names):
        if name in train_dict.keys():
            shutil.copy(path, os.path.join(train_dir, name))
            write_line(t_train_path, train_dict[name])
        elif name in val_dict.keys():
            shutil.copy(path, os.path.join(val_dir, name))
            write_line(t_val_path, val_dict[name])
        elif name in test_dict.keys():
            shutil.copy(path, os.path.join(test_dir, name))
            write_line(t_test_path, test_dict[name])


if __name__ == '__main__':
    main()
