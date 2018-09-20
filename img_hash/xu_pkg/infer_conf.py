# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:56:38 2018
@author: liuxu
"""
import numpy as np
from easydict import EasyDict as edict
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
config = edict()
# network related params
config.PIXEL_MEANS = np.array([103., 116., 123.])
config.SCALES = (224, 224)  # first is scale (the shorter side); second is max size

config.LABEL_DICT = {'0': 'music',
                     '1': 'dancing',
                     '2': 'pets',
                     '3': 'level1',
                     '4': 'diy',
                     '5': 'photograph',
                     '6': 'level1',
                     '7': 'food',
                     '8': 'others'}

config.LEVEL1_MODEL = ['level1', '14']
config.MUSIC_MODEL = ['music', '14']
config.DANCING_MODEL = ['dancing', '14']
config.PETS_MODEL = ['pets', '14']
config.DIY_MODEL = ['diy', '14']
config.PHOTOGRAPH_MODEL = ['photograph', '14']
config.FOOD_MODEL = ['food', '14']
config.OTHERS_MODEL = ['others', 14]
