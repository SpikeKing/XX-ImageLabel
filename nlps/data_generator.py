#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/13

生成验证数据
"""
import csv

from nlps.city_keywords import CHN_CITY_LIST, WORLD_CITY_LIST
from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def convert_chinese(word_list):
    res_list = []
    for word in word_list:
        # res_list.append(word.decode('gb2312'))
        res_list.append(word)
    return res_list


def filter_spaces(s):
    return s.replace('\n', '').replace('\t', '').replace('\r', '')


def get_all_regions():
    return list(CHN_CITY_LIST.keys()) + unfold_nested_list(CHN_CITY_LIST.values()) \
           + list(WORLD_CITY_LIST.keys()) + unfold_nested_list(WORLD_CITY_LIST.values())


def read_csv(file_name):
    lines = []  # 输出的行

    with open(file_name, encoding='gb2312') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        included_cols = [0, 1, 14, 18]  # ["ID", "缩略图", "标签", "描述"]
        count = 0
        r_count = 0  # 地域总数
        for row in csv_reader:
            count += 1
            if count == 1:  # 第1行是标题
                continue
            c_row = convert_chinese(row)  # 转换为中文

            if len(c_row) < 14 or not c_row[9]:  # 去掉无效的行
                continue

            c_row = [filter_spaces(c_row[i]) for i in included_cols]  # 过滤信息

            tags = c_row[2].replace('/', '').split(',')

            r_tags = []
            for tag in tags:
                if tag in get_all_regions():
                    r_tags.append(tag)

            r_dict = dict()
            if r_tags:
                r_dict['id'] = c_row[0]
                r_dict['images'] = c_row[1]
                r_dict['tags'] = ','.join(r_tags)
                r_dict['content'] = c_row[3]
                lines.append(json.dumps(r_dict, ensure_ascii=False))
                r_count += 1

        print('数据总数 {}, 地域总数 {}'.format(count, r_count))
        # for line in lines:
        #     print(list_2_utf8(json.loads(line)))

    return lines


def generate_mysql_data():
    csv_name = os.path.join(TXT_DATA, 'hot_content-2018-09-12-15384621.csv')
    csv_output = os.path.join(TXT_DATA, 'test_data-2018-09-12.txt')
    lines = read_csv(csv_name)
    for line in lines:
        write_line(csv_output, unicode_str(line))


def main():
    generate_mysql_data()


if __name__ == '__main__':
    main()
