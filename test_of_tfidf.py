#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/15

TF-IDF算法，Term Frequency - Inverse Document Frequency，词频率-逆文本频率，即TF和IDF
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == "__main__":
    corpus = [u"来到 北京 清华大学",
              u"来到 优盛",
              u"硕士 毕业 北京",
              u"很爱 北京"]

    cv = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    model_cv = cv.fit_transform(corpus)  # 计算tf-idf

    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(model_cv)  # 将文本转为词频矩阵

    word = cv.get_feature_names()  # 获取词袋模型中的所有词语

    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j], weight[i][j]
