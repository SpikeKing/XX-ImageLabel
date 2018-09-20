#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/18
"""
import json

import mxnet as mx
from keras_preprocessing.image import save_img

from mxnet import gluon
from tensorflow.contrib.gan.python.eval import preprocess_image

from img_hash.ml_trainer import MultiLabelTrainer
from utils.project_utils import *


def main():
    # sym, arg_params, aux_params = mx.model.load_checkpoint("base_net", 0)
    # net.load_parameters('testnet.params')
    save_img()
    pass



if __name__ == '__main__':
    main()
