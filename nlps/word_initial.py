#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/12
"""
import os

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def add_new_word():
    kw_folder = os.path.join(TXT_DATA, 'res_kw', 'cities')
    raw_folder = os.path.join(TXT_DATA, 'raws', 'cities')
    kw_path_list, kw_name_list = traverse_dir_files(kw_folder)
    raw_path_list, raw_name_list = traverse_dir_files(raw_folder)

    same = set(raw_name_list) & set(kw_name_list)
    raw_diff = set(raw_name_list) - same
    kw_diff = set(kw_name_list) - same

    for raw_word in raw_diff:
        write_line(os.path.join(kw_folder, raw_word), raw_word)

    print(raw_diff)
    print(kw_diff)


def main():
    add_new_word()


if __name__ == '__main__':
    main()
