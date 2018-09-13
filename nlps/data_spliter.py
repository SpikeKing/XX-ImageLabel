#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/12
"""

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def process_raw_cities(file_name, out_folder):
    lines = read_file_utf8(file_name)
    path_list, name_list = traverse_dir_files(out_folder)

    name_ids_dict = dict()
    for path, name in zip(path_list, name_list):
        region_lines = read_file_utf8(path)
        for region_line in region_lines:
            rid, _, = region_line.split(',', 1)
            if name not in name_ids_dict:
                name_ids_dict[name] = set()
            name_ids_dict[name].add(rid)

    name_path_dict = dict(zip(name_list, path_list))

    n_tag = set()
    n_line = 0

    for line in lines:
        data = json.loads(line)
        l_id = data['id']
        l_tags = data['tags'].split(',')
        l_content = data['content']

        for tag in l_tags:
            if tag in name_list:
                tag_ids = name_ids_dict[tag]
                if tag not in tag_ids:
                    n_path = name_path_dict[tag]
                    tag_line = l_id + "," + l_content
                    print('新增line: {}'.format(tag_line))
                    n_line += 1
                    write_line(n_path, tag_line)
                else:
                    continue
            else:
                print('新增Tag: {}'.format(tag))
                n_tag.add(tag)
                n_path = os.path.join(out_folder, tag)
                tag_line = l_id + "," + l_content
                print('新增line: {}'.format(tag_line))
                n_line += 1
                write_line(n_path, tag_line)

    print('新增Tag数: {}, 新增Line数: {}'.format(len(n_tag), n_line))


if __name__ == '__main__':
    data_file = os.path.join(TXT_DATA, 'test_data-2018-09-12.txt')
    out_folder = os.path.join(TXT_DATA, 'raws', 'cities')
    process_raw_cities(data_file, out_folder)
