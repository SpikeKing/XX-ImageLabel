#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/19
"""
from contents.content_tags import CONTENT_TAGS
from img_hash.dir_const import DATA_DIR
from img_hash.xu_pkg.im_hash_api import ImageHash
from root_dir import ROOT_DIR
from utils.project_utils import *


def process_data():
    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    train_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    train_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别

    data_lines = read_file_utf8(train_file)

    class_names = list(CONTENT_TAGS.keys())
    pathes, names = traverse_dir_files(train_folder)

    ih = ImageHash()

    for path, name in zip(pathes, names):
        res = ih.predict(path)
        print(res)


def main():
    process_data()


if __name__ == '__main__':
    main()
