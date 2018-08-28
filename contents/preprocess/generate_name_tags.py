#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/21
"""

import os

from contents.content_tags import traverse_tags
from contents.dir_consts import KEYWORDS_DIR
from utils.project_utils import write_line, mkdir_if_not_exist


def write_tag_keywords():
    """
    写入文本的标签
    :return: None
    """
    kw_folder = KEYWORDS_DIR
    mkdir_if_not_exist(kw_folder)
    all_tags = traverse_tags()
    for tag in all_tags:
        file_name = os.path.join(kw_folder, tag)
        write_line(file_name, tag)  # 写入全部标签


def main():
    write_tag_keywords()


if __name__ == '__main__':
    main()
