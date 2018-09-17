#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/17
"""
from nlps.nlp_dir import TXT_DATA
from utils.project_utils import *

TIME_TAGS = {
    u'时段': [u'早上', u'中午', u'下午', u'深夜'],
    u'周': [u'周末', u'工作日'],
    u'季节': [u'秋天', u'冬天', u'春天', u'夏天'],
    u'法定节假日': [
        u'中秋节', u'国庆节', u'元旦', u'春节', u'清明节',
        u'劳动节', u'端午节'],
    u'传统节日': [u'七夕', u'除夕', u'元宵节', u'重阳节'],
    u'世界节日': [u'光棍节_双11', u'万圣节', u'感恩节', u'平安夜', u'圣诞节',
              u'情人节', u'白色情人节', u'父亲节', u'母亲节', u'愚人节'],
    u'学生时间': [u'上学时间', u'不上学时间']
}


def get_time_tags():
    tags = unfold_nested_list(TIME_TAGS.values())
    return tags


def main():
    tags = get_time_tags()
    print('标签数: {}'.format(len(tags)))
    times_dir = os.path.join(TXT_DATA, 'res_kw', 'times')
    _, time_tags = traverse_dir_files(times_dir)
    print('文件标签数: {}'.format(len(time_tags)))


if __name__ == '__main__':
    main()
