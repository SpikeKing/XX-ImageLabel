#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/4
"""
import os
import sys
from configparser import ConfigParser

import mxnet as mx
import numpy as np

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss, TripletLoss
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.nn import Dense
from mxnet.initializer import Xavier
from mxnet.ndarray import sigmoid

from img_hash.dir_const import DATA_DIR
from img_hash.ih_dataloader import MultilabelDataset, TripletDataset
from root_dir import ROOT_DIR
from utils.project_utils import safe_div


def get_base_net(ctx):
    """
    获取base net，默认是mobilenet v2
    :param ctx: 运行环境，cpu or gpu，默认cpu
    :return: base net，基础网络
    """
    base_net = get_model('mobilenet1.0', pretrained=True)
    with base_net.name_scope():
        base_net.output = Dense(units=27)  # 全连接层
    base_net.output.initialize(Xavier(), ctx=ctx)  # 初始化
    base_net.collect_params().reset_ctx(ctx)
    base_net.hybridize()
    return base_net


def get_batch_acc(outputs, labels):
    """
    multiLabel acc, all values are same, is right
    :param outputs: predictions
    :param labels: ground truth
    :return: acc percent, num of right, num of all
    """
    outputs = sigmoid(outputs)

    outputs = outputs.asnumpy()
    labels = labels.asnumpy().astype('int')

    print(labels[0])
    outputs = np.where(outputs > 0.5, 1, 0)  # 类别阈值0.5
    print(outputs[0])
    rights = np.sum(np.where(labels == outputs, 0, 1), axis=1)
    n_right = np.count_nonzero(rights == 0)  # 全0的即全部相等
    return safe_div(n_right, len(labels)), n_right, len(labels)


def get_data_path():
    """
    return image folder and img file, labels in file
    :return: folder path and file path
    """
    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    img_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
    img_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别
    return img_folder, img_file


def get_train_data(batch_size=8):
    """
    process train data, add transforms.
    :param batch_size: per process num of samples
    :return: train data
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_folder, img_file = get_data_path()
    td = MultilabelDataset(data_folder=img_folder, data_file=img_file)
    train_data = DataLoader(td.transform_first(transform_train), batch_size=batch_size, shuffle=True)
    return train_data


def get_triplet_train_data(batch_size=8):
    """
    triplet loss
    :param batch_size: 批次大小
    :return:
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_folder, img_file = get_data_path()
    img_saved = os.path.join(img_file + ".tp.npz")
    td = TripletDataset(data_folder=img_folder, data_file=img_file,
                        saved_path=img_saved, transform=transform_train)
    train_data = DataLoader(td, batch_size=batch_size, shuffle=True)
    return train_data


def get_configs():
    cf = ConfigParser()
    cf.read(os.path.join(ROOT_DIR, 'img_hash', 'ih_configs.conf'))
    is_gpu = cf.getboolean("net", "is_gpu")
    batch_size = cf.getint("net", "batch_size")
    return {'is_gpu': is_gpu, 'batch_size': batch_size}


def get_context(is_gpu):
    ctx = mx.gpu(0) if is_gpu > 0 else mx.cpu()
    return ctx


def train_model():
    epochs = 5

    configs = get_configs()
    is_gpu = configs['is_gpu']
    batch_size = configs['batch_size']
    ctx = get_context(is_gpu)

    print("gpu: {}, batch_size: {}".format(is_gpu, batch_size))

    base_net = get_base_net(ctx=ctx)

    trainer = Trainer(base_net.collect_params(), 'rmsprop', {'learning_rate': 1e-3})

    bc_loss = SigmoidBinaryCrossEntropyLoss()
    triplet_loss = TripletLoss(margin=0)

    train_data = get_train_data(batch_size=batch_size)  # train data
    triplet_train_data = get_triplet_train_data(batch_size=batch_size)  # 训练数据

    for epoch in range(epochs):
        train_loss = 0  # 训练loss
        total_right, total_all = 0, 0

        for i, (batch, tp_batch) in enumerate(zip(train_data, triplet_train_data)):
            data, labels = batch[0], batch[1].astype('float32')
            tp_data, tp_labels = tp_batch[0], tp_batch[1].astype('float32')

            # print(data.shape, labels.shape)
            # print(tp_data.shape, tp_labels.shape)

            data = data.as_in_context(context=ctx)
            labels = labels.as_in_context(context=ctx)
            tp_data = tp_data.as_in_context(context=ctx)
            tp_labels = tp_labels.as_in_context(context=ctx)

            tp_data = mx.nd.transpose(tp_data, (1, 0, 2, 3, 4))
            tp_labels = mx.nd.transpose(tp_labels, (1, 0, 2))

            # print(tp_data.shape, tp_labels.shape)

            anc_ins, pos_ins, neg_ins = tp_data[0, :], tp_data[1, :], tp_data[2, :]

            # print(anc_ins.shape, pos_ins.shape, neg_ins.shape)

            with autograd.record():
                outputs = base_net(data)
                v_bc_loss = bc_loss(outputs, labels)

                inter1 = base_net(anc_ins)
                inter2 = base_net(pos_ins)
                inter3 = base_net(neg_ins)
                v_triplet_loss = triplet_loss(inter1, inter2, inter3)  # 交叉熵

            autograd.backward([v_bc_loss, v_triplet_loss])
            trainer.step(batch_size)

            print('bc: {}, triplet: {}'.format(np.sum(v_bc_loss.asnumpy()), np.sum(v_triplet_loss.asnumpy())))

            train_loss += v_bc_loss.mean().asscalar()
            acc, nr, na = get_batch_acc(outputs, labels)
            total_right += nr
            total_all += na

            if i != 0:  # batch 0 doesn't have train_loss.
                print('batch: %s, loss: %s, acc: %s' % (i, train_loss / i, acc))
            else:
                print('batch: %s' % i)

        train_loss /= len(train_data)
        print('epoch: %s, loss: %s, acc: %s' % (epoch, train_loss, total_right / total_all))


def main():
    train_model()


if __name__ == '__main__':
    main()
