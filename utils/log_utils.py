#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/9
"""


def unicode_str(s):
    """
    将字符串转换为unicode
    :param s: 字符串
    :return: unicode字符串
    """
    if not isinstance(s, unicode):  # 转换为unicode编码
        s = unicode(s, "utf-8")
    return s


def print_info(log_str):
    """
    打印日志
    :param log_str: 日志信息
    :return: None
    """
    print('[Info] ' + str(log_str))


def print_ex(log_str):
    """
    打印日志
    :param log_str: 日志信息
    :return: None
    """
    print('[Exception] ' + str(log_str))


def print_info_u(log_str):
    """
    打印日志
    :param log_str: 日志信息
    :return: None
    """
    try:
        log_str = unicode_str(str(log_str))
    except:
        log_str = log_str
    print(u'[Info] ' + log_str)


def print_ex_u(log_str):
    """
    打印日志
    :param log_str: 日志信息
    :return: None
    """
    log_str = unicode_str(log_str)
    print(u'[Exception] ' + str(log_str))
