#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""

import csv
import os

from contents.content_tags import traverse_tags
from img_hash.dir_const import DATA_DIR
from utils.project_utils import write_line, mkdir_if_not_exist, create_file, unicode_list


def get_csv_reader(file_name, encoding='gb2312'):
    """
    读取gb2312文件, 文件行
    """
    rows = []
    with open(file_name, encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rows.append(row)
    return rows


def remove_slash(s):
    """
    过滤换行符
    """
    return s.replace('\n', '').replace('\t', '').replace('\r', '')


def filter_content_tags(tags):
    """
    根据末尾1, 确定新的标签体系
    :param tags: 待过滤标签列表
    :return: unicode的标签
    """
    r_tags = []
    for tag in tags:
        if tag.endswith('1'):
            r_tags.append(tag.replace('1', ''))
    return unicode_list(r_tags)


def process_csv(file_path):
    """
    处理CSV文件
    :param file_path: csv文件名
    :return: None
    """
    file_name = file_path.split('/')[-1]  # 文件名
    out_name = file_name.replace('.csv', '.txt')

    csv_rows = get_csv_reader(file_path)

    included_cols = [0, 1, 14, 18]  # ["ID", "标签", "描述"]
    tags_all = traverse_tags()

    out_file = os.path.join(DATA_DIR, out_name)
    create_file(out_file)

    count = 0
    for row in csv_rows:
        count += 1
        if count == 1:
            print(row)
        if count == 1 or not row or len(row) < 13:  # 去掉头部
            continue
        c_row = [remove_slash(row[i]) for i in included_cols]
        [c_id, c_imgs, r_tag, c_content] = c_row
        c_tags = filter_content_tags(r_tag.split(','))  # 只保留最终的Tag
        for c_tag in c_tags:
            if c_tag in tags_all:
                write_line(out_file, c_id + u'---' + c_imgs + u'---' + ','.join(c_tags) + u'---' + c_content)
                break

    print('CSV 处理完成!')


def main():
    file_name = os.path.join(DATA_DIR, 'hot_content-2018-08-30-18001412.csv')
    process_csv(file_name)


if __name__ == '__main__':
    main()
