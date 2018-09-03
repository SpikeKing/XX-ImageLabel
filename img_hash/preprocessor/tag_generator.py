#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/31
"""
import copy

from contents.content_tags import SUB_CONTENT_TAGS, CONTENT_TAGS
from img_hash.dir_const import DATA_DIR
from utils.project_utils import *


def reverse_dict(map_data):
    inv_map = dict()
    for k, vs in map_data.items():
        for v in vs:
            inv_map[v] = k
    return inv_map


def up_tags(c_tags):
    """
    将多维度的标签列表, 转换为2级列表
    :param c_tags: 标签
    :return: 数据
    """
    tags = set(copy.deepcopy(c_tags))  # 避免修改原始数据

    three_2_two_dict = reverse_dict(SUB_CONTENT_TAGS)
    two_2_one_dict = reverse_dict(CONTENT_TAGS)

    # 由3级变成2级
    two_tags = set()
    remove_tags = set()
    for tag in tags:
        up_tag = three_2_two_dict.get(tag, None)
        if up_tag:
            two_tags.add(up_tag)
            remove_tags.add(tag)

    tags = tags.difference(remove_tags)
    tags = tags | two_tags

    # 由2级变成1级
    two_tags = set()
    remove_tags = set()
    for tag in tags:
        up_tag = two_2_one_dict.get(tag, None)
        if up_tag:
            two_tags.add(up_tag)
            remove_tags.add(tag)

    tags = tags.difference(remove_tags)
    tags = tags | two_tags

    return list(tags)


def main():
    file_path = os.path.join(DATA_DIR, 'hot_content-2018-08-30-18001412.txt')
    out_path = os.path.join(DATA_DIR, 'img_tags.txt')
    create_file(out_path)
    lines = read_file_utf8(file_path)
    for line in lines:
        [cid, imgs, tags_str, _] = line.split('---', 3)
        tags = tags_str.split(',')
        f_tags = up_tags(tags)
        imgs_name = [img.split('/')[-1] for img in imgs.split(',')]
        for n_img in imgs_name:
            write_line(out_path, n_img + "---" + ','.join(f_tags) + "---" + cid)


if __name__ == '__main__':
    main()
