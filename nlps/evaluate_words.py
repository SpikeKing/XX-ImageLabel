#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""
import os
import string

import jieba

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def filter_punc_and_num(s):  # 过滤标点和数字
    ch_punctuation = [u'。', u'，', u'“', u'”', u'：', u'、', u'！', u'～', u'？', u'；']
    num_punctuation = u'1234567890'
    for c in string.punctuation:
        s = s.replace(c, u" ")
    for c in ch_punctuation:
        s = s.replace(c, u" ")
    for c in num_punctuation:
        s = s.replace(c, u" ")
    return s


def merge_spaces(content):
    return ' '.join(content.split())  # 合并空格


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def jieba_cut_long(content):
    res_list = []
    sub_cons = chunks(content, 100)
    for sub_con in sub_cons:
        seg_list = jieba_cut(sub_con)  # 分词
        res_list += seg_list
    return res_list


def jieba_cut(content):
    seg_list = jieba.cut(content)  # 分词
    seg_list = [elem for elem in seg_list if elem.strip()]  # 去除全部空格
    return seg_list


def load_all_files(all_file):
    data_lines = read_file(all_file)
    res_dict = dict()
    for data_line in data_lines:
        [cid, _, content] = data_line.split('---', 2)
        res_dict[cid] = content
    return res_dict


def evaluate_words(words, tag_file, all_file):
    res_dict = load_all_files(all_file)

    tag_name = tag_file.split('/')[-1]
    lines = read_file(tag_file)
    n_count = len(lines)
    print('Tag: {} - {}'.format(tag_name, n_count))

    r_count = 0
    for data_line in lines:
        cid, content = data_line.split(',', 1)
        content = filter_punc_and_num(content.decode('utf8'))
        content = merge_spaces(content)  # 将多个空格合并一个

        all_count = 0
        for word_item in words:
            all_count += content.count(word_item)

        if all_count != 0:
            r_count += 1
        res_dict.pop(cid, None)
        # print(u'{}: {}'.format(cid, all_count))

    print('准确率: {:.4f}'.format(safe_div(r_count, n_count)))

    w_count = 0
    for key in res_dict.keys():
        content = res_dict[key]
        content = content.decode('utf8')
        all_count = 0
        for word_item in words:
            all_count += content.count(word_item)
        if all_count != 0:
            print(u'{}: {}'.format(key, all_count))
            print(content)
            w_count += 1

    print('错误数: {}'.format(w_count))


def main():
    tag_file = os.path.join(TXT_DATA, 'raws', '天津')
    all_file = os.path.join(TXT_DATA, 'all_raws')
    evaluate_words(words=[u'天津', u'民园广场'], tag_file=tag_file, all_file=all_file)


if __name__ == '__main__':
    main()
