#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/27
"""
import os

import mxnet as mx
import numpy as np

from mxnet import autograd, gluon, symbol, image, nd
from mxnet.gluon import SymbolBlock
from mxnet.gluon.data.vision import transforms
from mxnet.ndarray import sigmoid

from contents.content_tags import CONTENT_TAGS
from img_hash.dir_const import DATA_DIR


class MLPredictor(object):
    def __init__(self):
        self.net_path_cl = os.path.join(DATA_DIR, 'model', 'epoch-99-0.50-20180921002904.params-symbol.json')
        self.params_path_cl = os.path.join(DATA_DIR, 'model', 'epoch-99-0.50-20180921002904.params-0099.params')
        self.net_path_tl = os.path.join(
            DATA_DIR, 'model', 'tripletloss-epoch-99-0.74-20180922195150.params-symbol.json')
        self.params_path_tl = os.path.join(
            DATA_DIR, 'model', 'tripletloss-epoch-99-0.74-20180922195150.params-0099.params')
        self.net_cl, self.net_tl = self.load_net()
        self.ctx = mx.cpu()

    @staticmethod
    def class_names():
        return list(CONTENT_TAGS.keys())

    def load_net(self):
        net_cl = SymbolBlock.imports(self.net_path_cl, ['data'], self.params_path_cl)
        net_tl = SymbolBlock.imports(self.net_path_tl, ['data'], self.params_path_tl)
        return net_cl, net_tl

    def detect_img_to_class(self, img_path):
        img = image.imread(img_path)

        transform_fn = transforms.Compose([
            transforms.Resize(224, keep_ratio=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img = transform_fn(img)
        img = nd.expand_dims(img, axis=0)

        res = self.net_cl(img.as_in_context(self.ctx))
        res = sigmoid(nd.squeeze(res)).asnumpy()
        res = np.where(res > 0.5, 1, 0)
        indexes, = np.where(res == 1)
        return indexes

    def detect_img_to_vector(self, img_path):
        img = image.imread(img_path)

        transform_fn = transforms.Compose([
            transforms.Resize(224, keep_ratio=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img = transform_fn(img)
        img = nd.expand_dims(img, axis=0)
        res = self.net_tl(img.as_in_context(self.ctx)).asnumpy()
        return res

    # def detect_img_to_hash(self, img_path):
    #     detect_img_to_vector()
    #     return res


def test_of_MLPredictor():
    img_path = os.path.join(DATA_DIR, 'imgs_data', 'd4YE10xHdvbwKJV5yBYsoJJke6K9b.jpg')
    mlp = MLPredictor()

    clazzes = mlp.detect_img_to_class(img_path)
    res_classes = [mlp.class_names()[i] for i in clazzes.tolist()]
    print('测试类别: {}'.format(res_classes))

    vectors = mlp.detect_img_to_vector(img_path)
    print('相似向量: {}'.format(vectors))


def main():
    test_of_MLPredictor()


if __name__ == '__main__':
    main()
