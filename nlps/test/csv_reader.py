#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""

import csv
import os
import string

import jieba

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import write_line, mkdir_if_not_exist, read_file, create_file


def read_csv(file_name):
    lines = []
    tag_dict = dict()
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        included_cols = [0, 9, 13]  # ["ID", "标签", "描述"]
        count = 0
        for row in csv_reader:
            count += 1
            if count == 1:
                continue
            c_row = convert_word(row)
            if len(c_row) < 14 or not c_row[9]:  # 去掉错误的行
                continue
            c_row = [c_row[i].replace('\n', '').replace('\t', '').replace('\r', '') for i in included_cols]
            lines.append(c_row)
            tags = c_row[1].replace('/', '').split(',')
            for tag in tags:
                if tag not in tag_dict:
                    tag_dict[tag] = []
                tag_dict[tag].append((c_row[0], c_row[2]))

        print('count {}'.format(count))
    return lines, tag_dict


def process_csv(file_name):
    data_lines, tag_dict = read_csv(file_name)

    all_file = os.path.join(TXT_DATA, 'all_raws')
    create_file(all_file)

    for data_line in data_lines:
        cid, tags, content = data_line
        write_line(all_file, cid + u'---' + tags + u'---' + content)
        # seg_list = cut_sentence(content)
        # print(seg_list)
        # if seg_list:
        #     write_line(all_file, ' '.join(seg_list))

    tags_folder = os.path.join(TXT_DATA, 'raws')
    mkdir_if_not_exist(tags_folder, is_delete=True)
    for tag in tag_dict.keys():
        tag_file = os.path.join(tags_folder, tag)
        feed_dict = dict()
        for data_feed in tag_dict[tag]:
            (feed_id, content) = data_feed
            if feed_id in feed_dict:
                print('重复 ID {}'.format(feed_id))
            feed_dict[feed_id] = content
        for feed_id in feed_dict.keys():
            content = feed_dict[feed_id]
            write_line(tag_file, feed_id + u',' + content)
            # seg_list = cut_sentence(content)
            # if seg_list:
            #     write_line(tag_file, ' '.join(seg_list))


def load_stopwords():
    sw_path = os.path.join(TXT_DATA, 'stop_words')
    stop_words = read_file(sw_path)
    return stop_words


def cut_sentence(content):
    raw_sw = load_stopwords()
    content = test_repl(content)
    seg_list = jieba.cut(content, cut_all=True)
    sw_list = []
    for sw in raw_sw:
        sw = sw.decode('UTF-8')
        sw_list.append(sw)

    res_seg = []
    for seg in seg_list:
        if seg in sw_list or seg == u' ':
            continue
        res_seg.append(seg)
    return res_seg


def test_repl(s):  # From S.Lott's solution
    ch_punctuation = [u'。', u'，', u'“', u'”', u'：', u'、', u'！', u'～', u'？']
    num_punctuation = '1234567890'
    for c in string.punctuation:
        s = s.replace(c, "")
    for c in ch_punctuation:
        s = s.replace(c, "")
    for c in num_punctuation:
        s = s.replace(c, "")
    return s


def convert_word(word_list):
    res_list = []
    for word in word_list:
        res_list.append(word.decode('gb2312'))
    return res_list


def main():
    file_name = os.path.join(TXT_DATA, 'hot_content-2018-08-08-17283268.csv')
    process_csv(file_name)


if __name__ == '__main__':
    main()
