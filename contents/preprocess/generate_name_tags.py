#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/21
"""
from itertools import chain

import os

from contents.content_tags import CONTENT_TAGS, SUB_CONTENT_TAGS
from contents.dir_consts import KEYWORDS_DIR
from utils.project_utils import unicode_str, write_line


def unicode_list(data_list):
    """
    将list转换为unicode list
    :param data_list: 数量列表
    :return: unicode列表
    """
    return [unicode_str(s) for s in data_list]


def traverse_tags():
    """
    遍历全部标签
    :return: 标签列表
    """
    one_level = unicode_list(list(CONTENT_TAGS.keys()))
    two_level = unicode_list(list(chain.from_iterable(CONTENT_TAGS.values())))

    ot_error = set(one_level) & set(two_level)
    if ot_error:
        raise Exception('一级和二级标签重复!')

    one_two_level = one_level + two_level
    three_level = unicode_list(list(chain.from_iterable(SUB_CONTENT_TAGS.values())))
    oth_error = set(one_two_level) & set(three_level)
    if oth_error:
        raise Exception('三级标签重复! %s' % oth_error)

    all_tags = one_two_level + three_level
    print('标签数量: {}'.format(len(all_tags)))

    return all_tags


def write_tag_keywords():
    """
    写入文本的标签
    :return: None
    """
    kw_folder = KEYWORDS_DIR
    all_tags = traverse_tags()
    for tag in all_tags:
        file_name = os.path.join(kw_folder, tag)
        write_line(file_name, tag)  # 写入全部标签


def main():
    write_tag_keywords()


if __name__ == '__main__':
    main()
