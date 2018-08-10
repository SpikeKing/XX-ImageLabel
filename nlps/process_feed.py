#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""
from nlps.city_keywords import CHN_CITY_LIST, WORLD_CITY_LIST
from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def remove_verbose_words(content):
    verbose_words = ["威尼斯水城", "东方威尼斯", "桂林米粉", "长沙发", "墨西哥餐厅", "越南菜", "台湾腔", "马来西亚朋友",
                     "德国啤酒", "巴黎铁塔", "纽约INS网黄游乐场", "香港黑帮老大", "英国的Space NK", "日本料理", "武汉菜",
                     "布拉格的这本生活方式杂志", "意大利餐厅", "日本限定新品体验会", "土耳其菜", "土耳其餐厅", "新加坡餐厅",
                     "台湾美食"]
    for v_word in verbose_words:
        content = content.replace(v_word, "")
    return content


def load_words():
    kw_path = os.path.join(TXT_DATA, 'res_kw', 'cities')
    path_list, name_list = traverse_dir_files(kw_path)
    city_word_dict = dict()
    word_city_dict = dict()
    for path, city in zip(path_list, name_list):
        words = read_file(path)
        city_word_dict[city] = words
        for word in words:
            if not word:
                continue
            if word not in word_city_dict:
                word_city_dict[word] = []
            word_city_dict[word].append(city)
    return city_word_dict, word_city_dict


def get_key_from_val(value, v_dict):
    for v_key, v_value in v_dict.iteritems():  # for name, age in list.items():  (for Python 3.x)
        if value in v_value:
            return v_key
    return None


def get_sub_cities(city_name):
    city_name = city_name.decode('utf8')
    if city_name in CHN_CITY_LIST:
        return CHN_CITY_LIST[city_name]
    if city_name in WORLD_CITY_LIST:
        return WORLD_CITY_LIST[city_name]
    return []


def word_2_city(word_list, word_city_dict):
    city_list = []
    for word in word_list:
        city_name = word_city_dict[word]
        city_list.append(city_name)
    return city_list


def output_one_tag(c_indexes, c_cities):
    c_indexes, c_cities = sort_two_list(c_indexes, c_cities)
    up_cities_set = CHN_CITY_LIST.keys() + WORLD_CITY_LIST.keys()
    up_cities, sub_cities = [], []  # 一级地名, 二级地名
    up_indexes, sub_indexes = [], []
    for c_city, c_index in zip(c_cities, c_indexes):
        c_city = c_city[0]
        if c_city.decode('utf8') in up_cities_set:
            up_cities.append(c_city)
            up_indexes.append(c_index)
        else:
            sub_cities.append(c_city)
            sub_indexes.append(c_index)

    def_city = c_cities[0][0]

    if not up_cities or not sub_cities:
        return def_city

    up_city = up_cities[0]
    sub_city = sub_cities[0]

    if def_city == sub_city:
        return sub_city
    elif def_city == up_city:
        if sub_city.decode('utf8') in get_sub_cities(up_city):
            return sub_city
        else:
            return up_city
    else:
        return def_city


def process_feed(tag_file, all_file):
    tag_name = tag_file.split('/')[-1]
    sub_cities = get_sub_cities(tag_name)

    feed_lines = read_file(tag_file)
    all_lines = read_file(all_file)
    n_count = len(feed_lines)
    print('Tag: {} - {}'.format(tag_name, n_count))

    city_word_dict, word_city_dict = load_words()
    word_list = word_city_dict.keys()
    print('单词数: {}'.format(len(word_list)))

    r_count = 0
    tag_ids = set()
    for feed_line in feed_lines:
        cid, content = feed_line.split(',', 1)
        content = remove_verbose_words(content)
        c_words = []
        c_index = []
        for k_word in word_list:
            index = content.find(k_word)
            n_index = content.count(k_word)
            if index != -1 and n_index != 0:
                c_words.append(k_word)
                c_index.append((1.0 / float(n_index), index))

        if not c_index or not c_words:
            print(content)
            print(list_2_utf8(c_words))
            print(c_index)
            print('-' * 30)
            continue

        tag_ids.add(cid)

        c_index, c_words = sort_two_list(c_index, c_words)
        c_cities = word_2_city(c_words, word_city_dict)
        # show_string(c_cities)
        res_tag = output_one_tag(c_index, c_cities)  # 输出1个Tag

        if sub_cities and res_tag.decode('utf8') in sub_cities:  # 子集
            res_tag = tag_name

        if res_tag == tag_name:
            r_count += 1
        else:
            print(content)
            print(list_2_utf8(c_words))
            print(c_index)
            print('-' * 30)

    print(r_count)
    print('召回率: {:0.2f} %'.format(safe_div(r_count, n_count) * 100.0))

    error_tag = os.path.join(TXT_DATA, 'error_tag')
    if safe_div(r_count, n_count) < 1.0:
        write_line(error_tag, u'{},{}'.format(tag_name.decode('utf8'), safe_div(r_count, n_count)))

    w_count = 0
    for line in all_lines:
        try:
            aid, tags, content = line.split('---', 2)
            content = remove_verbose_words(content)
        except:
            print(line)
            continue

        if aid in tag_ids:
            continue
        c_words = []
        c_index = []
        for k_word in word_list:
            index = content.find(k_word)
            n_index = content.count(k_word)

            if index != -1 and n_index != 0:
                c_words.append(k_word)
                c_index.append((1.0 / float(n_index), index))

        if not c_index or not c_words:
            continue

        c_index, c_words = sort_two_list(c_index, c_words)

        res_tag = word_city_dict[c_words[0]][0]
        if res_tag == tag_name:
            w_count += 1
            # print(aid)
            # print(content)
            # print(list_2_utf8(c_words))
            # print(c_index)
            # print('-' * 30)

    print('精准率: {:0.2f} %'.format(safe_div(r_count, w_count + r_count) * 100.0))


def main():
    # tag_file = os.path.join(TXT_DATA, 'raws', 'cities', '上海')
    file_list, name_list = traverse_dir_files(os.path.join(TXT_DATA, 'raws', 'cities'))
    # print(len(file_list))
    all_file = os.path.join(TXT_DATA, 'all_raws')
    for tag_file in file_list:
        process_feed(tag_file=tag_file, all_file=all_file)
    print('-' * 50)


if __name__ == '__main__':
    main()
