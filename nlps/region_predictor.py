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
from utils.project_utils import read_file, unicode_str, traverse_dir_files, sort_two_list, safe_div


class RegionPredictor(object):
    KEY_WORD_FOLDER = os.path.join(TXT_DATA, 'res_kw', 'cities')
    EX_WORDS_FILE = os.path.join(TXT_DATA, 'res_kw', 'cities_other', 'ex_words')

    def __init__(self):
        self.cw_dict, self.wc_dict = self.__load_cities_words()
        self.ex_words = self.__load_ex_words()

    @staticmethod
    def __filter_pnc_and_num(s):
        """
        过滤数字和中文符号
        :param s: 字符串
        :return: 过滤后的字符串
        """
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
    def __load_ex_words():
        """
        加载需要排除的词汇，转换为unicode
        :return: 需要排除的词汇
        """
        ex_words = read_file(RegionPredictor.EX_WORDS_FILE)
        ex_words = [unicode_str(w) for w in ex_words]
        return ex_words

    @staticmethod
    def __load_cities_words():
        """
        加载城市的词汇
        :return: 当前的词汇
        """
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
                # print_ex_u(u'重复词汇: {}'.format(wc_key))
                # print_ex_u(u'出现于 %s' % list_2_utf8(words))
                raise Exception(u'重复词汇, 请删除!')
            else:
                wc_dict[wc_key] = words[0]
        # print_info('词汇数: %s, 城市数: %s' % (len(wc_dict.keys()), len(cw_dict.keys())))
        return cw_dict, wc_dict

    @staticmethod
    def __merge_spaces(content):
        """
        合并多个空格
        :param content: 内容
        :return: 合并后的内容
        """
        return u' '.join(content.split())  # 合并空格

    def __remove_ex_words(self, content):
        """
        删除排除的词汇
        :param content: 内容
        :return: 排除词汇的内容
        """
        for v_word in self.ex_words:
            content = content.replace(v_word, u"")
        return content

    @staticmethod
    def get_sub_cities(city_name):
        """
        获得子城市
        :param city_name: 省份
        :return: 城市
        """
        if city_name in CHN_CITY_LIST:
            return CHN_CITY_LIST[city_name]
        if city_name in WORLD_CITY_LIST:
            return WORLD_CITY_LIST[city_name]
        return []

    @staticmethod
    def analyze_cities(c_weights, c_cities):
        """
        分析，权重和城市
        :param c_weights: 权重
        :param c_cities: 城市
        :return: 选择最优的城市
        """
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
        # print(u'默认: {}'.format(def_city))
        # print(u'一级: {}, 二级: {}'.format(list_2_utf8(up_cities), list_2_utf8(sub_cities)))

        if not up_cities or not sub_cities:  # 只有一个直接返回默认
            return def_city

        up_city = up_cities[0]
        sub_city = sub_cities[0]
        # print(u'一级: {}, 二级: {}'.format(up_city, sub_city))

        up_subs = RegionPredictor.get_sub_cities(up_city)  # 获取子集

        if def_city == sub_city:  # 如果为1级则返回
            return sub_city
        elif def_city == up_city:  # 如果为1级
            if sub_city in up_subs:  # 判断2级是不是子类
                return sub_city  # 返回2级
            else:
                return up_city
        else:
            return def_city

    @staticmethod
    def __get_final_cities(regions):
        """
        根据城市的一级和二级
        :param regions: 城市列表
        :return: 当前城市
        """

        r_up, r_sub = [], []
        for region in regions:  # 真实
            region = unicode_str(region)
            if region in CHN_CITY_LIST.keys() + WORLD_CITY_LIST.keys():
                r_up.append(region)
            for sub_cv in CHN_CITY_LIST.values() + WORLD_CITY_LIST.values():
                if region in sub_cv:
                    r_sub.append(region)

        r1_final = []
        if r_sub:
            r1_final = r_sub
        elif r_up:
            for r1 in r_up:
                sub_cities = RegionPredictor.get_sub_cities(r1)
                if sub_cities:
                    r1_final += sub_cities
                else:
                    r1_final += r1

        return r1_final

    @staticmethod
    def is_tags_equal(real_dict, pred_dict):
        """
        判断标签是否相等
        :param real_dict: 真实字典
        :param pred_dict: 预测字典
        :return: 是否相等
        """
        r_regions = real_dict.get('regions', None)
        p_regions = pred_dict.get('regions', None)

        r1_final = RegionPredictor.__get_final_cities(r_regions)  # 真实城市
        r2_final = RegionPredictor.__get_final_cities(p_regions)  # 预测城市

        if not r1_final and not r2_final:  # 全无是正确
            return True
        elif not r1_final and r2_final:  # 真实没有，预测有，是正确
            return True
        elif r1_final and not r2_final:  # 真实有，预测没有，是错误
            return False
        return set(r2_final).issubset(r1_final)  # 返回子集

    def predict(self, content):
        """
        预测内容的城市
        :param content: 内容
        :return: 城市字典
        """
        content = unicode_str(content)
        # print(u'{} {}'.format(type(content), content))
        # cid, content = content.split(',', 1)
        # print(u'{} {}'.format(cid, content))
        content = self.__filter_pnc_and_num(content)
        # print(u'{}'.format(content))
        content = self.__merge_spaces(content)
        # print(u'{}'.format(content))
        content = self.__remove_ex_words(content)
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
        # print(u'{} {}'.format(list_2_utf8(c_cities), list_2_utf8(c_weights)))
        if not c_weights or not c_cities:  # 没有关键词
            return {}
        r_city = self.analyze_cities(c_weights, c_cities)
        print(u'最终城市: {}'.format(r_city))
        res_dict = {"regions": [r_city]}
        return res_dict  # 返回城市


def test_is_equal():
    rp = RegionPredictor()
    print(rp.is_tags_equal({"regions": [u'中国', u'海南']}, {"regions": [u'中国', u'三亚']}))  # True
    print(rp.is_tags_equal({"regions": []}, {"regions": [u'中国', u'厦门']}))  # True
    print(rp.is_tags_equal({"regions": [u'中国', u'海南']}, {"regions": []}))  # False
    print(rp.is_tags_equal({"regions": [u'中国', u'海南', u'福建']}, {"regions": ['三亚']}))  # True
    print(rp.is_tags_equal({"regions": [u'中国', u'海南', u'福建']}, {"regions": ['三亚', '广州']}))  # False
    print(rp.is_tags_equal({"regions": [u'中国', u'海南', u'福建']}, {"regions": ['三亚', '厦门']}))  # True
    print(rp.is_tags_equal({"regions": [u'世界', u'意大利']}, {"regions": ['罗马']}))  # True
    print(rp.is_tags_equal({"regions": [u'世界', u'意大利']}, {"regions": ['巴黎']}))  # False
    print(rp.is_tags_equal({"regions": [u'世界', u'法国']}, {"regions": ['巴黎']}))  # False
    print(rp.is_tags_equal({"regions": [u'日本']}, {"regions": ['日本']}))  # True


def main():
    rp = RegionPredictor()
    rp.predict(u'三亚最适合潜水的海岛，终于来啦分界洲岛位于海南岛的东南方向是三亚最适宜潜水、观赏海底世界的海岛，也被很多人称为')
    # '是“心灵的分界岛”、“坠落红尘的天堂”、“一个可以发呆的地方” 乘船过渡单程大概需要15分钟左右岛上有座红白相间的标志物'
    # '灯塔景色特别美，远处有山海水虽然已经挺深了，但依旧是蓝的特别清透，站在岸上都能看到很多热带鱼和螃蟹，因为能见度很高，'
    # '所以海岛的潜水是个不能错过的项目凹岛上的客流量相对于不是很多，所以不要担心人挤人哈哈，一定要坐观光车来到山顶，一览'
    # '岛上的风景从观景台看悬崖峭壁和大海我很爱听海浪拍打在岩石上的声音，可以让人安静下来。因为当天时间很赶，到达岛上'
    # '已经是下午了，晚上的离岛时间是六点所以很多地方还没有去，不过已经很满足了还有其他美丽的景点和有趣的地方就等你们来'
    # '继续解锁啦#暑假来这浪#')


if __name__ == '__main__':
    test_is_equal()
