#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/18
"""
import json
import mxnet as mx

from utils.project_utils import *


def save_mx_net(net, out_file):
    """
    保存的网络类型为 HybridSequential
    参考: https://discuss.mxnet.io/t/save-cnn-model-architecture-and-params/683/2
    :param net: 网络
    :param out_file: 输出文件
    :return: None
    """
    sym_json = net(mx.sym.var('data')).tojson()  # 转换为JSON字符串
    write_line(out_file, sym_json)
