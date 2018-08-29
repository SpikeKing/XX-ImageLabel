#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/28
"""
import os
import string

from contents.dir_consts import *
from utils.project_utils import *


class ContentPredictor(object):
    KEY_WORDS_DIR = KEYWORDS_DIR

    def __init__(self):
        self.tw_dict, self.wt_dict = self.__load_tag_words()

    @staticmethod
    def __read_lines(file_name, encoding='utf8'):
        """
        读取gb2312文件, 文件行
        """
        with open(file_name, encoding=encoding) as kf:
            rows = kf.readlines()
        return [r.strip() for r in rows]

    @staticmethod
    def __merge_spaces(content):
        """
        合并多个空格
        :param content: 内容
        :return: 合并后的内容
        """
        return u' '.join(content.split())  # 合并空格

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
    def __load_tag_words():
        path_list, name_list = traverse_dir_files(ContentPredictor.KEY_WORDS_DIR)
        tw_dict, wt_dict = dict(), dict()  # Tag->词; 词->Tag

        for path, city in zip(path_list, name_list):
            city = unicode_str(city)
            words = ContentPredictor.__read_lines(path)  # 读取单词列表
            tw_dict[city] = [unicode_str(w) for w in words]
            for word in tw_dict[city]:
                word = unicode_str(word)
                if not word:  # 去除空行
                    continue
                if word not in wt_dict:
                    wt_dict[word] = []
                wt_dict[word].append(city)

        for wc_key in wt_dict:  # 检测重复词汇, 将词与城市对应
            words = wt_dict[wc_key]
            if len(words) != 1:
                # print_ex_u(u'重复词汇: {}'.format(wc_key))
                # print_ex_u(u'出现于 %s' % list_2_utf8(words))
                print(words)
                print(wc_key)
                raise Exception(u'重复词汇, 请删除!')
            else:
                wt_dict[wc_key] = words[0]
        # print_info('词汇数: %s, 城市数: %s' % (len(wc_dict.keys()), len(cw_dict.keys())))
        return tw_dict, wt_dict

    def predict(self, content):
        content = unicode_str(content)  # 转换unicode

        content = self.__filter_pnc_and_num(content)  # 过滤标点和数字
        content = self.__merge_spaces(content)  # 合并空格
        print(content)
        word_list = self.wt_dict.keys()  # 全部词汇
        print('词数: {}'.format(len(word_list)))



def test_of_ContentPredictor():
    cp = ContentPredictor()
    cp.predict(u'七夕不知道送什么，就送口红啊雕牌这只变色唇膏可以说很红了，几乎人手一只如果你的妹子唇色浅，'
               u'不知道送什么就可以送这个如果唇色深和老干妈一样，建议还是不要了，妹子会恨你的！#七夕生存指南#')


if __name__ == '__main__':
    test_of_ContentPredictor()
