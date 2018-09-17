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
        time_tags = data_tags.get('times', None)
        if not time_tags:
            continue

        out_line = did + "---" + ",".join(time_tags) + "---" + content
        write_line(res_file, out_line)
        for time_tag in time_tags:
            tag_count[time_tag] += 1

        if count == 1000:
            print(count)

    for time_k in TIME_TAGS.keys():
        print(time_k)
        for time_v in TIME_TAGS[time_k]:
            print('\t{}: {}'.format(time_v, tag_count.get(time_v, 0)))
    print(tag_count)


def main():
    process_data()


if __name__ == '__main__':
    main()
