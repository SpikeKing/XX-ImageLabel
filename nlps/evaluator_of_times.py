#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/17
"""
import os

from nlps.nlp_dir import TXT_DATA
from nlps.tags_of_time import TIME_TAGS
from nlps.tags_predictor import TagPredictor
from utils.project_utils import *


def process_data():
    raw_file = os.path.join(TXT_DATA, 'test_data-2018-09-12.all.txt')
    data_lines = read_file_utf8(raw_file)

    res_file = os.path.join(TXT_DATA, 'times_cases.txt')
    create_file(res_file)

    tp = TagPredictor()

    tag_count = collections.defaultdict(int)

    print('样本总数: {}'.format(len(data_lines)))

    for count, data_line in enumerate(data_lines):
        data_dict = json.loads(data_line)
        did = data_dict['id']
        content = data_dict.get('content', None)
        if not content:
            continue
        try:
            data_tags = tp.predict(content)
        except Exception as e:
            print(content)
            raise Exception('Error')
        time_tags = data_tags.get('times', None)
        time_detail = tp.get_time_detail(content)
        if not time_tags:
            continue

        out_line = did + "---" + ",".join(time_tags) + "---" + time_detail + "---" + content
        write_line(res_file, out_line)
        for time_tag in time_tags:
            tag_count[time_tag] += 1

        if count % 1000 == 0:
            print('count: {}'.format(count))

    for time_k in TIME_TAGS.keys():
        print(time_k + " - ", end='')
        for time_v in TIME_TAGS[time_k]:
            print('{}:{}  '.format(time_v, tag_count.get(time_v, 0)), end='')

    print(tag_count)
    print('时间标签: {}'.format(len(read_file_utf8(res_file))))


def main():
    process_data()
    # raw_file = os.path.join(TXT_DATA, 'test_data-2018-09-12.all.txt')
    # res_file = os.path.join(TXT_DATA, 'times_cases.txt')
    # data_lines1 = read_file_utf8(raw_file)
    # data_lines2 = read_file_utf8(res_file)
    # print(len(data_lines1), len(data_lines2), safe_div(len(data_lines2), len(data_lines1)))


if __name__ == '__main__':
    main()
