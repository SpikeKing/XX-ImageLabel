#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *


def read_words(file_name):
    lines = read_file(file_name)
    one_line = ''
    for line in lines:
        one_line += line + ' '
    return one_line


def tfidf_main(target_line, all_line, name):
    corpus = [target_line, all_line]

    vectorizer = CountVectorizer(lowercase=False)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    weight1 = weight[0]
    print(len(weight1))
    print(len(words))
    weight1, words = sort_two_list(weight1, words, reverse=True)
    count = 0
    for word, w in zip(words, weight1):
        str_line = u'{},{:0.4f}'.format(unicode(word), w)
        write_line(name, str_line)
        # print(word)
        # print(w)
        count += 1
        if count == 10:
            break


def main():
    # file_name = os.path.join(TXT_DATA, 'tags', '土耳其')
    paths, names = traverse_dir_files(os.path.join(TXT_DATA, 'tags'))
    out_dir = os.path.join(TXT_DATA, 'keywords')
    for path, name in zip(paths, names):
        all_name = os.path.join(TXT_DATA, 'all')
        target_line = read_words(path)
        all_line = read_words(all_name)
        out_file = os.path.join(out_dir, name)
        tfidf_main(target_line, all_line, out_file)


if __name__ == "__main__":
    main()
    # load_stopwords()
