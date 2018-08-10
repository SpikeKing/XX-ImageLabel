#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/10
"""

import os
import string

from nlps.city_keywords import CHN_CITY_LIST, WORLD_CITY_LIST
from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


class RegionPredictor(object):
    KEY_WORD_FOLDER = os.path.join(TXT_DATA, 'res_kw', 'cities')
    EX_WORDS_FILE = os.path.join(TXT_DATA, 'res_kw', 'cities_other', 'ex_words')

    def __init__(self):
        self.cw_dict, self.wc_dict = self.load_cities_words()
        self.ex_words = self.load_ex_words()

    @staticmethod
    def filter_pnc_and_num(s):  # 过滤标点和数字
        ch_punctuation = [u'。', u'，', u'“', u'”', u'：', u'、', u'！', u'～', u'？', u'；']
        num_punctuation = u'1234567890'
        for c in string.punctuation:
            s = s.replace(c, u" ")
        for c in ch_punctuation:
            s = s.replace(c, u" ")
        for c in num_punctuation:
            s = s.replace(c, u" ")
        return s

    @staticmethod
    def load_ex_words():
        ex_words = read_file(RegionPredictor.EX_WORDS_FILE)
        ex_words = [unicode_str(w) for w in ex_words]
        return ex_words

    @staticmethod
    def load_cities_words():
        path_list, name_list = traverse_dir_files(RegionPredictor.KEY_WORD_FOLDER)
        cw_dict, wc_dict = dict(), dict()  # 城市->词; 词->城市
        for path, city in zip(path_list, name_list):
            city = unicode_str(city)
            words = read_file(path)  # 读取单词列表
            cw_dict[city] = words
            for word in words:
                word = unicode_str(word)
                if not word:  # 去除空行
                    continue
                if word not in wc_dict:
                    wc_dict[word] = []
                wc_dict[word].append(city)

        for wc_key in wc_dict:  # 检测重复词汇, 将词与城市对应
            words = wc_dict[wc_key]
            if len(words) != 1:
                RegionPredictor.print_ex(u'重复词汇: {}'.format(wc_key))
                RegionPredictor.print_ex(u'出现于 %s' % list_2_utf8(words))
                raise Exception(u'重复词汇, 请删除!')
            else:
                wc_dict[wc_key] = words[0]
        RegionPredictor.print_into(u'词汇数: %s, 城市数: %s' % (len(wc_dict.keys()), len(cw_dict.keys())))
        return cw_dict, wc_dict

    @staticmethod
    def print_into(s):
        print(u'[Info] %s' % s)

    @staticmethod
    def print_ex(s):
        print(u'[Exception] %s' % s)

    @staticmethod
    def merge_spaces(content):
        return u' '.join(content.split())  # 合并空格

    def remove_ex_words(self, content):
        for v_word in self.ex_words:
            content = content.replace(v_word, u"")
        return content

    @staticmethod
    def analyze_cities(c_weights, c_cities):
        c_weights, c_cities = sort_two_list(c_weights, c_cities)
        up_cities_set = CHN_CITY_LIST.keys() + WORLD_CITY_LIST.keys()
        up_cities, up_weights, sub_cities, sub_weights = [], [], [], []  # 一级地名, 二级地名

        for c_city, c_index in zip(c_cities, c_weights):
            c_city = c_city
            if c_city in up_cities_set:
                up_cities.append(c_city)
                up_weights.append(c_index)
            else:
                sub_cities.append(c_city)
                sub_weights.append(c_index)

        def_city = c_cities[0]
        print(u'默认: {}'.format(def_city))
        # print(u'一级: {}, 二级: {}'.format(list_2_utf8(up_cities), list_2_utf8(sub_cities)))

        if not up_cities or not sub_cities:  # 只有一个直接返回默认
            return def_city

        up_city = up_cities[0]
        sub_city = sub_cities[0]
        print(u'一级: {}, 二级: {}'.format(up_city, sub_city))

        # if def_city == sub_city:
        #     return sub_city
        # elif def_city == up_city:
        #     if sub_city.decode('utf8') in get_sub_cities(up_city):
        #         return sub_city
        #     else:
        #         return up_city
        # else:
        #     return def_city

    def predict(self, content):
        content = unicode_str(content)
        # print(u'{} {}'.format(type(content), content))
        cid, content = content.split(',', 1)
        # print(u'{} {}'.format(cid, content))
        content = self.filter_pnc_and_num(content)
        # print(u'{}'.format(content))
        content = self.merge_spaces(content)
        # print(u'{}'.format(content))
        content = self.remove_ex_words(content)
        # print(u'{}'.format(content))
        word_list = self.wc_dict.keys()  # 全部词汇
        c_words, c_weights = [], []  # 单词和权重
        for k_word in word_list:
            iw = content.find(k_word)  # 索引位置
            nw = content.count(k_word)  # 出现次数
            if iw != -1 and nw != 0:
                c_words.append(k_word)
                c_weights.append((safe_div(1, nw), iw))  # 1/的目的是改变单调性
        # print(u'{} {}'.format(list_2_utf8(c_words), list_2_utf8(c_weights)))
        c_cities = [unicode_str(self.wc_dict[w]) for w in c_words]
        print(u'{} {}'.format(list_2_utf8(c_cities), list_2_utf8(c_weights)))
        self.analyze_cities(c_weights, c_cities)


def main():
    rp = RegionPredictor()
    rp.predict('6430405880479220737,三亚最适合潜水的海岛，终于来啦分界洲岛位于海南岛的东南方向是三亚最适宜潜水、观赏海底世界的海岛，也被很多人称为'
               '是“心灵的分界岛”、“坠落红尘的天堂”、“一个可以发呆的地方” 乘船过渡单程大概需要15分钟左右岛上有座红白相间的标志物'
               '灯塔景色特别美，远处有山海水虽然已经挺深了，但依旧是蓝的特别清透，站在岸上都能看到很多热带鱼和螃蟹，因为能见度很高，'
               '所以海岛的潜水是个不能错过的项目凹岛上的客流量相对于不是很多，所以不要担心人挤人哈哈，一定要坐观光车来到山顶，一览'
               '岛上的风景从观景台看悬崖峭壁和大海我很爱听海浪拍打在岩石上的声音，可以让人安静下来。因为当天时间很赶，到达岛上'
               '已经是下午了，晚上的离岛时间是六点所以很多地方还没有去，不过已经很满足了还有其他美丽的景点和有趣的地方就等你们来'
               '继续解锁啦#暑假来这浪#')


if __name__ == '__main__':
    main()
