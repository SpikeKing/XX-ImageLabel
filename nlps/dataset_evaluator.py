#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/13

评估目前的数据集
"""
from nlps.nlp_dir import TXT_DATA
from nlps.region_predictor import RegionPredictor
from utils.log_utils import print_info_u
from utils.project_utils import *


def process_data():
    raw_folder = os.path.join(TXT_DATA, 'raws', 'cities')
    path_list, name_list = traverse_dir_files(raw_folder)
    print_info_u(u'标签数量: {}'.format(len(path_list)))

    rp = RegionPredictor()

    cities_dict = dict()
    data_count = 0  # 总的测试数据
    right_count = 0

    for path, name in zip(path_list, name_list):
        name = unicode_str(name)
        data_lines = read_file(path)

        r_count = 0  # 正确的数量
        a_count = len(data_lines)  # 总数

        for data_line in data_lines:
            cities = rp.predict(data_line)
            if not cities:  # 检测为空
                continue
            city = cities['regions'][0]
            sub_cities = rp.get_sub_cities(name)
            if city == name or city in sub_cities:
                r_count += 1
            # print(city)

        right_count += r_count
        data_count += a_count
        rate = safe_div(r_count, a_count) * 100
        # print_info_u(u'{} 正确率: {} %'.format(name, rate))
        cities_dict[name] = rate

    cities_list = sort_dict_by_value(cities_dict)
    print_info_u(u'测试数据条目: {}'.format(data_count))
    print_info_u(u'标签数量: {}'.format(len(cities_list)))
    sum_rate = 0.0
    for name, rate in cities_list:
        print_info_u(u'{} 正确率: {} %'.format(name, rate))
        sum_rate += rate

    print_info_u(u'平均准确率: {}, 标签个数: {}'.format(safe_div(right_count, data_count), len(cities_list)))


def main():
    process_data()


if __name__ == '__main__':
    main()
