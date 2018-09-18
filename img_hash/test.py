#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/18
"""
import json

import mxnet as mx

from mxnet import gluon

from img_hash.ml_trainer import MultiLabelTrainer
from utils.project_utils import *


def main():
    json_line = read_file('data.json', mode='one')
    net = MultiLabelTrainer().get_base_net()
    net.load_parameters('testnet.params')


if __name__ == '__main__':
    main()
