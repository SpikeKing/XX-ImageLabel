#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/11
"""

import matplotlib
from mxnet.gluon.data.vision import transforms

from img_hash.data_loader import TripletDataset

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

from mxnet.gluon.data import DataLoader

from img_hash.dir_const import DATA_DIR
from root_dir import ROOT_DIR
from utils.project_utils import *


def show_tags(td, data):
    tags_idxes = np.where(data == 1)[0]
    tags_names = td.get_class_names()
    print([tags_names[i] for i in tags_idxes])


def test_of_triplet_loss():
    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    img_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    img_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别
    img_saved = os.path.join(img_file + ".tp.npz")

    td = TripletDataset(data_folder=img_folder, data_file=img_file, saved_path=img_saved)
    for count, data in enumerate(td):
        # print(data[0][0])
        if count == 4:
            plt.subplot(131)
            plt.imshow(data[0][0].asnumpy())
            show_tags(td, data[1][0])
            plt.subplot(132)
            plt.imshow(data[0][1].asnumpy())
            show_tags(td, data[1][1])
            plt.subplot(133)
            plt.imshow(data[0][2].asnumpy())
            show_tags(td, data[1][2])
            plt.show()
            # print(data)
            break


def test_of_trans():
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomLighting(0.1),
        # transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    img_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    img_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别
    img_saved = os.path.join(img_file + ".tp.npz")

    td = TripletDataset(data_folder=img_folder, data_file=img_file,
                        saved_path=img_saved, transform=transformer)
    # td = TripletDataset(data_folder=img_folder, data_file=img_file, saved_path=img_saved)

    train_data = DataLoader(td, batch_size=4, shuffle=True)

    for count, data in enumerate(train_data):
        print('OK')
        imgs, labels = data[0], data[1]
        print(imgs.shape, labels.shape)
        if count == 0:
            plt.subplot(131)
            plt.imshow(imgs[0][0].asnumpy())
            plt.subplot(132)
            plt.imshow(imgs[0][1].asnumpy())
            plt.subplot(133)
            plt.imshow(imgs[0][2].asnumpy())
            plt.show()
            break


def main():
    test_of_trans()


if __name__ == '__main__':
    main()
