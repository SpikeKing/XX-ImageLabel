#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/31
"""
from img_hash.dir_const import DATA_DIR
from utils.project_utils import *


def main():
    data_file = os.path.join(DATA_DIR, 'hot_content-2018-08-30-18001412.txt')
    out_file = os.path.join(DATA_DIR, 'img_list-2018-08-30-18001412.txt')
    lines = read_file_utf8(data_file)
    for line in lines:
        try:
            [_, imgs_str, _, _] = line.split('---', 3)  # 只分3次
        except Exception as e:
            print(e)
            print(line)
            exit(0)
        img_list = imgs_str.split(',')
        for img_path in img_list:
            write_line(out_file, img_path)
    print('图片列表完成!')


if __name__ == '__main__':
    main()
