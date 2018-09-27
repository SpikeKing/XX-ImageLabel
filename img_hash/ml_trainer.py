#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/9/11
"""
import os
import sys

import mxnet as mx
import numpy as np

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from configparser import ConfigParser
from utils.project_utils import *

from mxnet import autograd, gluon, symbol, image, nd
from mxnet.gluon.utils import split_and_load
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.nn import Dense, SymbolBlock
from mxnet.initializer import Xavier
from mxnet.ndarray import sigmoid

from img_hash.data_loader import MultilabelDataset, safe_div, TripletDataset
from img_hash.dir_const import DATA_DIR

from contents.content_tags import CONTENT_TAGS
from root_dir import ROOT_DIR


class MultiLabelTrainer(object):

    def __init__(self):
        """
        构造器，配置来源于配置文件config
        """
        self.class_names = list(CONTENT_TAGS.keys())  # 全部类别, 27个类别

        self.ctx = None
        self.n_class = len(self.class_names)
        self.batch_size = -1
        self.epochs = -1

        self.config_path = os.path.join(ROOT_DIR, 'img_hash', 'configs.conf')
        self.load_config(self.config_path)

        up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))

        self.train_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'train_data_416')
        self.train_file = os.path.join(DATA_DIR, "t_img_tags_train.txt")  # 数据类别

        self.val_folder = os.path.join(up_folder, 'data_set', 'XX-ImageLabel', 'val_data_416')
        self.val_file = os.path.join(DATA_DIR, "t_img_tags_val.txt")  # 数据类别

    def get_base_net(self):
        """
        获取base net，默认是mobilenet v2
        :return: base net，基础网络
        """
        base_net = get_model('mobilenet1.0', pretrained=True)
        with base_net.name_scope():
            base_net.output = Dense(units=self.n_class)  # 全连接层
        base_net.output.initialize(Xavier(), ctx=self.ctx)  # 初始化
        base_net.collect_params().reset_ctx(self.ctx)
        base_net.hybridize()
        return base_net

    @staticmethod
    def get_context(is_gpu):
        if is_gpu:
            num_gpus = 2
        else:
            num_gpus = -1
        ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
        return ctx

    @staticmethod
    def get_configs(config_path):
        """
        读取训练的配置
        """
        cf = ConfigParser()
        cf.read(config_path)
        is_gpu = cf.getboolean("net", "is_gpu")  # GPU
        batch_size = cf.getint("net", "batch_size")  # BatchSize
        epochs = cf.getint("net", "epochs")
        return {'is_gpu': is_gpu, 'batch_size': batch_size, 'epochs': epochs}

    def load_config(self, config_path):
        """
        加载配置
        """
        configs = self.get_configs(config_path)
        is_gpu = configs['is_gpu']

        self.ctx = self.get_context(is_gpu)
        self.batch_size = configs['batch_size']
        self.epochs = configs['epochs']

    def get_train_data(self, batch_size):
        """
        获取训练数据，数据扩充
        """
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomLighting(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        td = MultilabelDataset(data_folder=self.train_folder, data_file=self.train_file,
                               transform=transform_train)
        train_data = DataLoader(dataset=td, batch_size=batch_size, shuffle=True)

        return train_data, len(td)

    def get_tl_train_data(self, batch_size):
        """
        获取TripletLoss训练数据, 一组3个, 数据扩充
        """
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomLighting(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        td = TripletDataset(data_folder=self.train_folder, data_file=self.train_file,
                            transform=transform_train, saved_path=True)
        train_data = DataLoader(dataset=td, batch_size=batch_size, shuffle=True)

        return train_data, len(td)

    def get_val_data(self, batch_size):
        """
        获取验证数据，数据扩充
        """
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        td = MultilabelDataset(data_folder=self.val_folder, data_file=self.val_file,
                               transform=transform_val)
        val_data = DataLoader(td, batch_size=batch_size, shuffle=True)

        return val_data, len(td)

    def get_tl_val_data(self, batch_size):
        """
        获取TripletLoss验证数据, 一组3个, 数据扩充
        """
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        td = TripletDataset(data_folder=self.val_folder, data_file=self.val_file,
                            transform=transform_val, saved_path=True)
        val_data = DataLoader(dataset=td, batch_size=batch_size, shuffle=True)

        return val_data, len(td)

    @staticmethod
    def print_info(s):
        print('[Info] {}'.format(s))

    @staticmethod
    def metric_of_rpf(y_pred, y_true):
        ar, ap, af1 = 0, 0, 0  # 汇总
        for yp, yt in zip(y_pred, y_true):
            yt = yt.asnumpy().astype('int')
            yp = sigmoid(yp).asnumpy()
            yp = np.where(yp > 0.5, 1, 0)

            idx_true = np.where(yt == 1)[0]
            idx_pred = np.where(yp == 1)[0]

            tp = set(idx_true) & set(idx_pred)
            r = safe_div(len(tp), len(idx_true))
            p = safe_div(len(tp), len(idx_pred))
            f1 = safe_div(2 * r * p, (r + p))

            ar += r
            ap += p
            af1 += f1

        ar /= len(y_pred)
        ap /= len(y_pred)
        af1 /= len(y_pred)

        return ar, ap, af1

    def get_batch_rpf(self, outputs, labels):
        """
        获取批次的rpf
        :param outputs: 输出
        :param labels: 真值
        :return: r, p, f1
        """
        br, bp, bf1 = 0, 0, 0
        for output, label in zip(outputs, labels):
            r, p, f1 = self.metric_of_rpf(output, label)
            br += r
            bp += p
            bf1 += f1

        sp_batch = len(outputs)

        br /= sp_batch
        bp /= sp_batch
        bf1 /= sp_batch

        return br, bp, bf1

    def val_net(self, net, val_data, len_vd):

        n_batch = int(len_vd / self.batch_size)
        self.print_info('训练 - 样本数:{}, 批次样本: {}, 批次数: {}'
                        .format(len_vd, self.batch_size, n_batch))

        e_r, e_p, e_f1 = 0, 0, 0

        for i, batch in enumerate(val_data):
            data, labels = batch[0], batch[1].astype('float32')

            data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)  # 多GPU
            labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)

            outputs = [net(X) for X in data]

            br, bp, bf1 = self.get_batch_rpf(outputs, labels)

            e_r += br
            e_p += bp
            e_f1 += bf1

            self.print_info('validation: batch: {}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                            .format(i, br, bp, bf1))

            n_batch = i + 1

        e_r /= n_batch
        e_p /= n_batch
        e_f1 /= n_batch

        self.print_info('validation: recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                        .format(e_r, e_p, e_f1))
        return e_r, e_p, e_f1

    @staticmethod
    def save_net_and_params(net, epoch, value, name=None):
        """
        存储网络和参数
        :param net: 网络
        :param epoch: epoch
        :param value: 值
        :param name: 标题名
        :return: None
        """
        cp_dir = os.path.join(DATA_DIR, 'checkpoints')
        mkdir_if_not_exist(cp_dir)
        epoch_params = os.path.join(
            cp_dir,
            '{}-epoch-{}-{:.2f}-{}.params'.format(name, epoch, value, get_current_time_str()))
        net.export(epoch_params, epoch=epoch)

    def train_model_for_ml(self):
        """
        训练模型, 多标签
        """
        base_net = self.get_base_net()  # 基础网络
        train_data, len_td = self.get_train_data(self.batch_size)  # 训练数据，按批次获取
        val_data, len_vd = self.get_val_data(self.batch_size)  # 训练数据，按批次获取

        trainer = Trainer(base_net.collect_params(), 'rmsprop', {'learning_rate': 1e-4})
        loss_func = SigmoidBinaryCrossEntropyLoss()

        lr_steps = [10, 20, 30, np.inf]  # 逐渐降低学习率
        lr_factor = 0.75
        lr_counter = 0

        n_batch = int(len_td / self.batch_size)

        self.print_info('训练 - 样本数:{}, 批次样本: {}, 批次数: {}'
                        .format(len_td, self.batch_size, n_batch))

        for epoch in range(self.epochs):

            if epoch == lr_steps[lr_counter]:  # 逐渐降低学习率
                trainer.set_learning_rate(trainer.learning_rate * lr_factor)
                lr_counter += 1

            e_loss, e_r, e_p, e_f1 = 0, 0, 0, 0  # epoch

            for i, batch in enumerate(train_data):

                data, labels = batch[0], batch[1].astype('float32')

                data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)
                labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)

                with autograd.record():  # 梯度求导
                    outputs = [base_net(X) for X in data]
                    bc_loss = [loss_func(yhat, y) for yhat, y in zip(outputs, labels)]

                for l in bc_loss:
                    l.backward()

                trainer.step(self.batch_size)

                batch_loss = sum([l.mean().asscalar() for l in bc_loss]) / len(bc_loss)  # batch的loss
                e_loss += batch_loss

                br, bp, bf1 = self.get_batch_rpf(outputs, labels)

                e_r += br
                e_p += bp
                e_f1 += bf1

                self.print_info('batch: {}, loss: {:.5f}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                                .format(i, batch_loss, br, bp, bf1))

                n_batch = i + 1  # 批次数

            e_loss /= n_batch
            e_r /= n_batch
            e_p /= n_batch
            e_f1 /= n_batch

            self.print_info('epoch: {}, loss: {:.5f}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                            .format(epoch, e_loss, e_r, e_p, e_f1))
            e_r, e_p, e_f1 = self.val_net(base_net, val_data, len_vd)

            self.save_net_and_params(base_net, epoch, e_f1, name='multilabel')  # 存储网络

    def evaluate_net(self, base_net, val_data):
        triplet_loss = gluon.loss.TripletLoss(margin=0)
        rate = 0.0
        sum_correct, sum_all = 0, 0

        for i, batch in enumerate(val_data):
            data, labels = batch[0], batch[1].astype('float32')

            data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)  # 多GPU
            labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)
            for X in data:
                anchor_ins, pos_ins, neg_ins = [], [], []
                for b_X in X:
                    anchor_ins.append(nd.expand_dims(b_X[0], axis=0))
                    pos_ins.append(nd.expand_dims(b_X[1], axis=0))
                    neg_ins.append(nd.expand_dims(b_X[2], axis=0))

                anchor_ins = nd.concatenate(anchor_ins, axis=0)
                pos_ins = nd.concatenate(pos_ins, axis=0)
                neg_ins = nd.concatenate(neg_ins, axis=0)

                inter1 = base_net(anchor_ins)
                inter2 = base_net(pos_ins)
                inter3 = base_net(neg_ins)
                loss = triplet_loss(inter1, inter2, inter3)  # TripletLoss
                n_correct = np.sum(np.where(loss == 0, 1, 0))
                sum_all += loss.shape[0]
                sum_correct += n_correct
            rate = safe_div(sum_correct, sum_all)
            self.print_info('验证Batch: {}, 准确率: {:.4f} ({} / {})'.format(i, rate, sum_correct, sum_all))
        rate = safe_div(sum_correct, sum_all)
        self.print_info('验证准确率: %.4f (%s / %s)' % (rate, sum_correct, sum_all))
        return rate

    def train_model_for_tl(self):
        """
        训练Triplet Loss模型
        :return: 当前模型
        """
        net_path = os.path.join(DATA_DIR, 'model', 'epoch-24-0.54-20180920182658.params-symbol.json')
        params_path = os.path.join(DATA_DIR, 'model', 'epoch-24-0.54-20180920182658.params-0024.params')
        hash_num = 128

        base_net = gluon.nn.SymbolBlock.imports(net_path, ['data'], params_path)
        with base_net.name_scope():
            base_net.output = Dense(units=hash_num)  # 全连接层
        base_net.output.initialize(Xavier(), ctx=self.ctx)  # 初始化
        base_net.collect_params().reset_ctx(self.ctx)

        train_data, train_len = self.get_tl_train_data(self.batch_size)
        val_data, val_len = self.get_tl_val_data(self.batch_size)
        self.print_info("Triplet Loss 训练样本数: {}".format(train_len))
        self.print_info("Triplet Loss 验证样本数: {}".format(val_len))

        triplet_loss = gluon.loss.TripletLoss(margin=10.0)
        trainer = Trainer(base_net.collect_params(), 'rmsprop', {'learning_rate': 1e-4})

        for epoch in range(self.epochs):
            e_loss, final_i = 0, 0
            for i, batch in enumerate(train_data):
                data, labels = batch[0], batch[1].astype('float32')
                data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)
                labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)
                data_loss = []

                with autograd.record():  # 梯度求导
                    for X in data:
                        anchor_ins, pos_ins, neg_ins = [], [], []
                        for b_X in X:
                            anchor_ins.append(nd.expand_dims(b_X[0], axis=0))
                            pos_ins.append(nd.expand_dims(b_X[1], axis=0))
                            neg_ins.append(nd.expand_dims(b_X[2], axis=0))
                        anchor_ins = nd.concatenate(anchor_ins, axis=0)
                        pos_ins = nd.concatenate(pos_ins, axis=0)
                        neg_ins = nd.concatenate(neg_ins, axis=0)

                        inter1 = base_net(anchor_ins)
                        inter2 = base_net(pos_ins)
                        inter3 = base_net(neg_ins)

                        loss = triplet_loss(inter1, inter2, inter3)  # TripletLoss
                        data_loss.append(loss)

                for l in data_loss:
                    l.backward()

                curr_loss = np.mean([mx.nd.mean(loss).asscalar() for loss in data_loss])
                self.print_info("batch: {}, loss: {}".format(i, curr_loss))
                e_loss += curr_loss
                final_i = i + 1
                trainer.step(self.batch_size)

            self.print_info("epoch: {}, loss: {}".format(epoch, safe_div(e_loss, final_i)))
            dist_acc = self.evaluate_net(base_net, val_data)  # 评估epoch的性能
            self.save_net_and_params(base_net, epoch, dist_acc, name='tripletloss')  # 存储网络

    def test_model_for_ml(self):
        net_path = os.path.join(DATA_DIR, 'model', 'epoch-3-0.48-20180920164709.params-symbol.json')
        params_path = os.path.join(DATA_DIR, 'model', 'epoch-3-0.48-20180920164709.params-0003.params')

        net = gluon.nn.SymbolBlock.imports(net_path, ['data'], params_path)

        im_path = os.path.join(DATA_DIR, 'imgs_data', 'd4YE10xHdvbwKJV5yBYsoJJke6K9b.jpg')
        img = image.imread(im_path)

        # plt.imshow(img.asnumpy())
        # plt.show()

        transform_fn = transforms.Compose([
            transforms.Resize(224, keep_ratio=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img = transform_fn(img)

        img = nd.expand_dims(img, axis=0)
        res = net(img.as_in_context(self.ctx[0]))
        res = sigmoid(nd.squeeze(res)).asnumpy()
        res = np.where(res > 0.5, 1, 0)
        indexes, = np.where(res == 1)
        res_classes = [self.class_names[i] for i in indexes.tolist()]
        # print(indexes.tolist())
        print('测试类别: {}'.format(res_classes))


if __name__ == '__main__':
    mlt = MultiLabelTrainer()
    mlt.train_model_for_tl()
