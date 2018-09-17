#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/14
"""
import os

from root_dir import ROOT_DIR
from utils.project_utils import *

up_folder = os.path.join(ROOT_DIR, 'nlps', 'data_merger', 'data')
wcl_folder = os.path.join(ROOT_DIR, 'nlps', 'data_merger', 'data', 'txt_data_wcl')
zl_folder = os.path.join(ROOT_DIR, 'nlps', 'data_merger', 'data', 'txt_data_zl')


def merge_error_ids():
    wcl_error_ids = os.path.join(wcl_folder, 'res_kw', 'error_ids')
    zl_error_ids = os.path.join(zl_folder, 'res_kw', 'error_ids')
    out_file = os.path.join(up_folder, 'error_ids')
    create_file(out_file)

    # print(wcl_error_ids)
    wcl_lines = read_file_utf8(wcl_error_ids)
    zl_lines = read_file_utf8(zl_error_ids)

    res_dict = dict()
    for wcl_line in wcl_lines:
        if not wcl_line:
            continue
        city_name, id = wcl_line.split(',', 1)
        if city_name not in res_dict:
            res_dict[city_name] = set()
        res_dict[city_name].add(id)

    for zl_line in zl_lines:
        if not zl_line:
            continue
        city_name, id = zl_line.split(',', 1)
        if city_name not in res_dict:
            res_dict[city_name] = set()
        res_dict[city_name].add(id)

    city_keys = sorted(list(res_dict.keys()))
    for city_name in city_keys:
        cids = res_dict[city_name]
        cids = sorted(list(cids))
        for cid in cids:
            write_line(out_file, city_name + ',' + cid)


def ex_words_merge():
    wcl_ex_ids = os.path.join(wcl_folder, 'res_kw', 'cities_other', 'ex_words')
    zl_ex_ids = os.path.join(zl_folder, 'res_kw', 'cities_other', 'ex_words')
    out_file = os.path.join(up_folder, 'ex_words')

    wcl_lines = read_file_utf8(wcl_ex_ids)
    zl_lines = read_file_utf8(zl_ex_ids)
    create_file(out_file)

    res_set = set()
    for wcl_line in wcl_lines:
        if not wcl_line:
            continue
        res_set.add(wcl_line)

    for zl_line in zl_lines:
        if not zl_line:
            continue
        res_set.add(zl_line)

    words = sorted(list(res_set))
    for word in words:
        write_line(out_file, word)


def merge_cities():
    wcl_city_folder = os.path.join(wcl_folder, 'res_kw', 'cities')
    zl_city_folder = os.path.join(zl_folder, 'res_kw', 'cities')
    out_folder = os.path.join(up_folder, 'cities')
    create_folder(out_folder)

    z_paths, z_names = traverse_dir_files(zl_city_folder)
    w_paths, w_names = traverse_dir_files(wcl_city_folder)

    print(len(z_names), len(w_names))

    for z_path, z_name, w_path, w_name in zip(z_paths, z_names, w_paths, w_names):
        if z_name != w_name:
            print('error')
            return
        w_words = read_file_utf8(w_path)
        z_words = read_file_utf8(z_path)

        final_words = set()
        for word in w_words:
            if not word:
                continue
            final_words.add(word)

        for word in z_words:
            if not word:
                continue
            final_words.add(word)

        final_words = sorted(list(final_words))

        for word in final_words:
            final_path = os.path.join(out_folder, z_name)
            write_line(final_path, word)


def main():
    # merge_cities()
    merge_error_ids()
    ex_words_merge()


if __name__ == '__main__':
    main()
