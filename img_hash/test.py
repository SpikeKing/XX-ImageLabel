#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/18
"""
import json

import mxnet as mx

from mxnet import gluon
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.nn import Dense
from mxnet.initializer import Xavier

from img_hash.ml_trainer import MultiLabelTrainer
from utils.project_utils import *


def main():
    base_net = get_model('mobilenet1.0', pretrained=True)
    base_net.features = base_net.features[62:64]
    base_net.features.add(Dense(units=128))
    print(base_net.features)
    base_net.features[-1].initialize(Xavier(), ctx=mx.cpu(0))  # 初始化
    base_net.collect_params().reset_ctx(mx.cpu(0))


if __name__ == '__main__':
    main()
