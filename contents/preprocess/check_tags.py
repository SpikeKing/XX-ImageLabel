#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/28
"""
from contents.content_tags import traverse_tags
from contents.dir_consts import SAMPLES_DIR
from utils.project_utils import *


def main():
    tags = traverse_tags()
    print('已有tags: {}'.format(len(tags)))
    paths, names = traverse_dir_files(SAMPLES_DIR)
    print('文件数: {}'.format(len(names)))


if __name__ == '__main__':
    main()
