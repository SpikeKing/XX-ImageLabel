#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:30:20 2018
@author: liuxu
"""
import os
import cv2
import numpy as np
import mxnet as mx
import math
import json
import sys
import random, shutil
import importlib

from img_hash.xu_pkg.infer_conf import config

cur_path = os.path.abspath(os.path.dirname(__file__))


class MeipaiRetrival(object):
    def __init__(self, *args, **kwargs):
        self.height = int(config.SCALES[0])
        self.width = int(config.SCALES[1])
        self.mean_pixels = config.PIXEL_MEANS

    def _load_module(self, model_name):
        prefix = os.path.join(self.root_path, model_name[0])
        epoch = model_name[1]
        sym, args, auxs = mx.model.load_checkpoint(prefix, int(epoch))
        if model_name[0] == 'level1':
            binary_hashing_layer = sym.get_internals()['binary_hashing_output']
            feature_layer = sym.get_internals()['relu5_5_sep_output']
            sym = mx.sym.Group([sym, binary_hashing_layer, feature_layer])
            data_shapes = [1, 5, 3, 224, 224]
        else:
            sym = sym.get_internals()['binary_hashing_output']
            data_shapes = [1, 5, 512, 14, 14]
        mod = mx.module.Module(
            symbol=sym,
            context=self.devs,
            data_names=('data',),
            label_names=None)
        mod.bind(data_shapes=[('data', data_shapes)],
                 label_shapes=None,
                 for_training=False)
        mod.set_params(args, auxs, allow_missing=True, allow_extra=True)
        return mod

    def load_model(self, rootpath, device_id=None, batch_size=1, *args, **kwargs):
        self.root_path = rootpath
        self.batch_size = batch_size

        if device_id is None:
            self.devs = mx.cpu()
        else:
            self.devs = [mx.gpu(int(i)) for i in device_id.split(',')]
        self.level1_mod = self._load_module(config.LEVEL1_MODEL)
        self.music_mod = self._load_module(config.MUSIC_MODEL)
        self.dancing_mod = self._load_module(config.DANCING_MODEL)
        self.diy_mod = self._load_module(config.DIY_MODEL)
        self.photograph_mod = self._load_module(config.PHOTOGRAPH_MODEL)
        self.food_mod = self._load_module(config.FOOD_MODEL)
        self.pets_mod = self._load_module(config.PETS_MODEL)
        self.others_mod = self._load_module(config.OTHERS_MODEL)

    def _img_preprocessiong(self, img):
        try:
            if len(img.shape) != 3 or img.shape[2] != 3:
                # input frame must be 3 channel RGB image
                return 502
        except:
            return 502
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img.reshape([1, 3, self.height, self.width])
        return img

    def _get_hashing_code(self, hashing_layer_output):
        hashing_layer_output[np.where(hashing_layer_output >= 0.5)] = 1
        hashing_layer_output[np.where(hashing_layer_output < 0.5)] = 0
        hashing_layer_output = list(hashing_layer_output.astype('int8'))
        #        hashing_layer_output = [str(x) for x in hashing_layer_output]
        #        hashing_code = int(''.join(hashing_layer_output),2)
        return hashing_layer_output  # hashing_code

    def predict(self, frame_list):
        frame_list = frame_list * 5
        if len(frame_list) != 5:
            # frame_list must contain 5 frames
            return [510]

        # frame_list = map(self._img_preprocessiong, frame_list)
        frame_list = [self._img_preprocessiong(f) for f in frame_list]
        for frame in frame_list:
            if isinstance(frame, int):
                return [502]
        #        print(frame_list[0].shape)
        frame_list = np.vstack(frame_list).reshape([1, 5, 3, 224, 224])
        self.level1_mod.forward(mx.io.DataBatch(data=[mx.nd.array(frame_list)]), is_train=False)

        softmax_output, _, _, _, hashing_layer, feature = self.level1_mod.get_outputs()

        softmax_output = softmax_output.asnumpy()
        KEY = np.argmax(softmax_output)
        which_type = config.LABEL_DICT[str(KEY)]
        if which_type == 'level1':
            hashing_layer = hashing_layer.asnumpy()
            hashing_code = hashing_layer[0]  # self._get_hashing_code(hashing_layer[0])
        else:
            eval('self.' + which_type + '_mod').forward(mx.io.DataBatch(data=[mx.nd.array(feature)]), is_train=False)
            hashing_layer = eval('self.' + which_type + '_mod').get_outputs()[0].asnumpy()
            hashing_code = hashing_layer[0]  # self._get_hashing_code(hashing_layer[0])
        return [KEY, hashing_code]

    def version(self, ):
        '''
        version for online update.
        equal to the version number on cf.
        '''
        return "20180723"


def get_plugin_class():
    return MeipaiRetrival
