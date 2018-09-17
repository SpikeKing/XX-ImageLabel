#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/10
"""

import os
import string

from nlps.tags_of_city import CHN_CITY_LIST, WORLD_CITY_LIST
from nlps.tags_of_time import TIME_TAGS
from nlps.nlp_dir import TXT_DATA

from utils.project_utils import *


class TagPredictor(object):
    KEY_WORD_FOLDER = os.path.join(TXT_DATA, 'res_kw', 'cities')
    TIME_KW_FOLDER = os.path.join(TXT_DATA, 'res_kw', 'times')
    EX_WORDS_FILE = os.path.join(TXT_DATA, 'res_kw', 'cities_other', 'ex_words')
    KEY_WORDS_FILE = os.path.join(TXT_DATA, 'res_kw', 'cities_other', 'key_words')
    S_KEY_WORDS_FILE = os.path.join(TXT_DATA, 'res_kw', 'cities_other', 's_key_words')

    def __init__(self):
        self.cw_dict, self.wc_dict = self.__load_tags_words(TagPredictor.KEY_WORD_FOLDER, True)
        self.tw_dict, self.wt_dict = self.__load_tags_words(TagPredictor.TIME_KW_FOLDER, False)
        self.ex_words = self.__load_ex_words()
        self.key_words = self.__load_key_words()  # 核心词
        self.s_key_words = self.__load_s_key_words()  # 核心词

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
            s = s.replace(c, u"")  # 数字直接去除
        return s

    @staticmethod
    def __load_ex_words():
        """
        加载需要排除的词汇，转换为unicode
        :return: 需要排除的词汇
        """
        ex_words = read_file_utf8(TagPredictor.EX_WORDS_FILE)
        ex_words = [unicode_str(w) for w in ex_words]
        return ex_words

    @staticmethod
    def __load_key_words():
        """
        关键词。转换为unicode
        :return: 需要排除的词汇
        """
        key_words = read_file_utf8(TagPredictor.KEY_WORDS_FILE)
        key_words = [unicode_str(w) for w in key_words]
        return key_words

    @staticmethod
    def __load_s_key_words():
        """
        关键词。转换为unicode
        :return: 需要排除的词汇
        """
        key_words = read_file_utf8(TagPredictor.S_KEY_WORDS_FILE)
        key_words = [unicode_str(w) for w in key_words]
        return key_words

    @staticmethod
    def __load_tags_words(file_name, is_ex=True):
        """
        加载标签的词汇
        :return: 当前的词汇
        """
        # path_list, name_list = traverse_dir_files(RegionPredictor.KEY_WORD_FOLDER)
        path_list, name_list = traverse_dir_files(file_name)
        cw_dict, wc_dict = dict(), dict()  # 标签->词; 词->标签
        for path, city in zip(path_list, name_list):
            city = unicode_str(city)
            words = read_file_utf8(path)  # 读取单词列表
            cw_dict[city] = words
            for word in words:
                word = unicode_str(word)
                if not word:  # 去除空行
                    continue
                if word not in wc_dict:
                    wc_dict[word] = []
                wc_dict[word].append(city)
        for wc_key in wc_dict:  # 检测重复词汇, 将词与标签对应
            words = wc_dict[wc_key]
            if len(words) != 1:
                print(u'重复词汇: {}'.format(wc_key))
                print(u'出现于 %s' % words)
                if is_ex:
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
        up_cities_set = list(CHN_CITY_LIST.keys()) + list(WORLD_CITY_LIST.keys())
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

        if sub_cities and sub_weights:
            for c, (sc, sw) in enumerate(zip(sub_cities, sub_weights)):
                for uc, uw in zip(up_cities, up_weights):
                    up_subs = TagPredictor.get_sub_cities(uc)  # 获取子集
                    if sc in up_subs:
                        (s, w1) = sw
                        (u, w2) = uw
                        sw = (1 / (1 / s + 1 / u), min(w1, w2))
                        sub_weights[c] = sw

            sub_weights, sub_cities = sort_two_list(sub_weights, sub_cities)
        # print(u'默认: {}'.format(def_city))
        # print(u'一级: {}, 二级: {}'.format(list_2_utf8(up_cities), list_2_utf8(sub_cities)))
        if not up_cities or not sub_cities:  # 只有一个直接返回默认
            return def_city
        up_city = up_cities[0]
        sub_city = sub_cities[0]
        # print(u'一级: {}, 二级: {}'.format(up_city, sub_city))
        up_subs = TagPredictor.get_sub_cities(up_city)  # 获取子集
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
                sub_cities = TagPredictor.get_sub_cities(r1)
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
        r1_final = TagPredictor.__get_final_cities(r_regions)  # 真实城市
        r2_final = TagPredictor.__get_final_cities(p_regions)  # 预测城市
        if not r1_final and not r2_final:  # 全无是正确
            return True
        elif not r1_final and r2_final:  # 真实没有，预测有，是正确
            return True
        elif r1_final and not r2_final:  # 真实有，预测没有，是错误
            return False
        return set(r2_final).issubset(r1_final)  # 返回子集

    def find_key_content(self, content):
        """
        根据关键词，找到关键内容
        :param content: 关键内容
        :return: 关键词
        """
        content = unicode_str(content)
        content = self.__remove_ex_words(content)  # 删除歧义词汇

        words = self.key_words
        s_words = self.s_key_words

        content_key = u""
        b_len = 5
        for word in words:
            # key_idx = content.findall(word)  # 关键部分
            key_idxes = []
            for match in re.finditer(str(word), content):
                key_idxes.append(match.start())
            if key_idxes:  # 关键词出现多次
                # print('info: {}'.format(word))
                for key_idx in key_idxes:
                    idx_len = min(len(content) - key_idx, 50)
                    start_idx = max(0, key_idx - b_len)
                    content_key += content[start_idx:start_idx + idx_len + b_len] + u" "

        for word in s_words:
            key_idxes = []
            for match in re.finditer(str(word), content):
                key_idxes.append(match.start())
            if key_idxes:  # 关键词出现多次
                # print('info: {}'.format(word))
                for key_idx in key_idxes:
                    start_idx = max(0, key_idx - 10)
                    end_idx = min(len(content), key_idx + 10)
                    content_key += content[start_idx:end_idx] + u" "

        return content_key

    def predict(self, content, is_debug=False):
        """
        预测内容
        :param content: 内容
        :param is_debug: 是否Debug
        :return: 标签
        """
        content_key = self.find_key_content(content)
        res_dict_key = self._predict(content_key, is_debug)
        if res_dict_key.get('regions', None):
            return res_dict_key
        regions_dict = self._predict(content, is_debug)
        times_dict = self._predict_time(content)

        res_dict = {'regions': regions_dict.get('regions', []),
                    'times': times_dict.get('times', [])}

        # print(res_dict)
        return res_dict

    def convert_tags(self, c_words, c_weights, word_dict):
        """
        将关键词转换为城市
        :param c_words: 关键词
        :param c_weights: 权重
        :return: 城市和权重
        """
        c_cities = []
        if not c_words or not c_weights:  # 异常，直接返回空
            return c_cities, c_weights
        c_words, c_weights = sort_two_list(c_words, c_weights)  # 排序
        # c_cities = [unicode_str(self.wc_dict[w]) for w in c_words]  # 将词转换为城市
        c_cities = [unicode_str(word_dict[w]) for w in c_words]  # 将词转换为城市

        remove_words = []  # 删除词汇
        # 获取需要删除的词汇
        for word1, weight1, city1 in zip(c_words, c_weights, c_cities):
            for word2, weight2, city2 in zip(c_words, c_weights, c_cities):
                if word1 == word2:
                    continue
                elif (word1 in word2) and (city1 != city2):  # 去除不同城市的包含词，如南京西路和南京
                    remove_words.append(word1)
        # 删除包含词
        t_words, t_weights, t_cities = [], [], []
        for word, weight, city in zip(c_words, c_weights, c_cities):
            if word in remove_words:
                continue
            if isinstance(city, list):  # 可能一个词对应两个标签
                for c in city:
                    t_words.append(word)
                    t_weights.append(weight)
                    t_cities.append(c)
            else:
                t_words.append(word)
                t_weights.append(weight)
                t_cities.append(city)
        city_dict = dict()  # 城市集
        for weight, city in zip(t_weights, t_cities):  # 词
            if city not in city_dict:
                city_dict[city] = weight  # 初始化
                continue
            num = 1. / weight[0]
            pos = weight[1]
            c_weight = city_dict[city]
            c_num = 1. / c_weight[0]
            c_pos = c_weight[1]
            city_dict[city] = (1. / (num + c_num), min(pos, c_pos))  # 更新相同城市的词
        city_dict_list = sort_dict_by_value(city_dict, reverse=False)  # 重排序
        r_cities, r_weights = [], []  # 有序的词汇
        for (city, weight) in city_dict_list:
            r_cities.append(city)
            r_weights.append(weight)
        return r_cities, r_weights

    def _predict(self, content, is_debug=False):
        """
        预测内容的城市，核心
        :param content: 内容
        :param is_debug: 调试
        :return: 城市字典
        """
        content = unicode_str(content)  # 转换unicode
        # print(u'{} {}'.format(type(content), content))
        # cid, content = content.split(',', 1)
        # print(u'{} {}'.format(cid, content))
        content = self.__filter_pnc_and_num(content)  # 过滤标点和数字
        # print(u'{}'.format(content))
        content = self.__merge_spaces(content)  # 合并空格
        # print(u'{}'.format(content))
        content = self.__remove_ex_words(content)  # 删除歧义词汇
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
        c_cities, c_weights = self.convert_tags(c_words, c_weights, self.wc_dict)  # 处理核心词汇
        # print(u'{} {}'.format(list_2_utf8(c_cities), list_2_utf8(c_weights)))
        if not c_weights or not c_cities:  # 没有关键词
            return {"regions": []}
        r_city = self.analyze_cities(c_weights, c_cities)
        # print(u'最终城市: {}'.format(r_city))
        res_dict = {"regions": [r_city]}
        if is_debug:
            res_dict["debug"] = zip(c_cities, c_weights)  # 增加debug信息
            return res_dict
        return res_dict  # 返回城市

    def _predict_time(self, content):
        """
        预测时间
        :param content: 内容
        :return: 标签字典
        """
        content = unicode_str(content)  # 转换unicode
        content = self.__filter_pnc_and_num(content)  # 过滤标点和数字
        content = self.__merge_spaces(content)  # 合并空格
        word_list = self.wt_dict.keys()  # 全部词汇
        c_words, c_weights = [], []  # 单词和权重

        for k_word in word_list:
            iw = content.find(k_word)  # 索引位置
            nw = content.count(k_word)  # 出现次数
            if iw != -1 and nw != 0:
                c_words.append(k_word)
                c_weights.append((safe_div(1, nw), iw))  # 1/的目的是改变单调性

        if not c_words:
            return {"times": []}

        c_times, c_weights = self.convert_tags(c_words, c_weights, self.wt_dict)  # 处理核心词汇

        final_tags = set()
        for t_type in TIME_TAGS.keys():
            for w_time in c_times:
                if w_time in TIME_TAGS[t_type]:
                    final_tags.add(w_time)  # 每个标签只添加一次
                    break
        final_tags = list(sorted(list(final_tags)))

        res_dict = {"times": final_tags}

        return res_dict

    def get_time_detail(self, content):
        content = unicode_str(content)  # 转换unicode
        content = self.__filter_pnc_and_num(content)  # 过滤标点和数字
        content = self.__merge_spaces(content)  # 合并空格
        word_list = self.wt_dict.keys()  # 全部词汇
        c_words, c_weights = [], []  # 单词和权重

        for k_word in word_list:
            iw = content.find(k_word)  # 索引位置
            nw = content.count(k_word)  # 出现次数
            if iw != -1 and nw != 0:
                c_words.append(k_word)
                c_weights.append((safe_div(1, nw), iw))  # 1/的目的是改变单调性

        if not c_words:
            return {"times": []}

        c_times, c_weights = self.convert_tags(c_words, c_weights, self.wt_dict)  # 处理核心词汇
        res_str = ""
        for c_word, c_weight in zip(c_words, c_weights):
            res_str += c_word + "-" + str(int(safe_div(1, c_weight[0]))) + ","
        if res_str:
            res_str = res_str[:len(res_str) - 1]  # 去掉最后的","
        return res_str


def test_is_equal():
    rp = TagPredictor()
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


def test_of_prediction():
    rp = TagPredictor()
    content = u'南沙天后宫，紧临珠江出海口伶仃洋，坐落于大角山东南麓，依山傍水，殿宇辉煌，建筑特点是集北京故宫的风格和南京中山陵的气势于一体，其规模是现今世界同类建筑之最，被誉为“天下天后第一宫”，也是东南亚最大的妈祖庙。殿中香烟袅袅地址广州市南沙区天后路88号#旅行#开放时间08:30~17:00门票价格20.00元L一路上可以在不同的高度远眺伶仃洋。使人视野开阔，亭台楼阁、一片大海，非常壮观，很漂亮。最后一幅图片是当时在那买的小兔子，很精致，很可爱。'
    res = rp.predict(content)
    print(json.dumps(res, ensure_ascii=False))
    print(rp.get_time_detail(content))


def test_of_time():
    rp = TagPredictor()
    # tw_dict, wt_dict = rp.load_times_words()
    # print(tw_dict)
    # print(wt_dict)


def main():
    test_of_prediction()


if __name__ == '__main__':
    main()
