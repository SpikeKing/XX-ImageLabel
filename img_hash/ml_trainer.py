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
from mxnet.gluon.utils import split_and_load

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from configparser import ConfigParser

from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.nn import Dense
from mxnet.initializer import Xavier
from mxnet.ndarray import sigmoid

from img_hash.data_loader import MultilabelDataset, safe_div
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
        # ctx = mx.gpu(0) if is_gpu > 0 else mx.cpu()  # 选择CPU或GPU
        if is_gpu:
            num_gpus = 3
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

        td = MultilabelDataset(data_folder=self.train_folder, data_file=self.train_file)
        train_data = DataLoader(td.transform_first(transform_train), batch_size=batch_size, shuffle=True)

        return train_data

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

        td = MultilabelDataset(data_folder=self.val_folder, data_file=self.val_file)
        val_data = DataLoader(td.transform_first(transform_val), batch_size=batch_size, shuffle=True)

        return val_data

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

    def val_net(self, net, val_data):
        e_r, e_p, e_f1 = 0, 0, 0
        self.print_info('验证批次数: {}'.format(len(val_data) / self.batch_size))

        final_i = 0
        for i, batch in enumerate(val_data):
            data, labels = batch[0], batch[1].astype('float32')
            data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)  # 多GPU
            labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)

            outputs = [net(X) for X in data]

            br, bp, bf1 = 0, 0, 0
            for output, label in zip(outputs, labels):
                r, p, f1 = self.metric_of_rpf(output, label)
                br += r
                bp += p
                bf1 += f1

            e_r += br
            e_p += bp
            e_f1 += bf1

            if i == 20:
                final_i = i
                break

            sp_batch = len(outputs)
            self.print_info('validation: batch: {}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                            .format(i, br / sp_batch, bp / sp_batch, bf1 / sp_batch))
            # r, p, f1 = self.metric_of_rpf(outputs, labels)
            # e_r += r
            # e_p += p
            # e_f1 += f1

        n_batch = final_i + 1
        e_r /= n_batch
        e_p /= n_batch
        e_f1 /= n_batch

        self.print_info('validation: recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                        .format(e_r, e_p, e_f1))

    def train_model(self):
        """
        训练模型
        """
        base_net = self.get_base_net()  # 基础网络
        train_data = self.get_train_data(self.batch_size)  # 训练数据，按批次获取
        val_data = self.get_val_data(self.batch_size)  # 训练数据，按批次获取

        trainer = Trainer(base_net.collect_params(), 'rmsprop', {'learning_rate': 1e-4})
        loss_func = SigmoidBinaryCrossEntropyLoss()

        lr_steps = [10, 20, 30, np.inf]  # 逐渐降低学习率
        lr_factor = 0.75
        lr_counter = 0

        for epoch in range(self.epochs):

            if epoch == lr_steps[lr_counter]:  # 逐渐降低学习率
                trainer.set_learning_rate(trainer.learning_rate * lr_factor)
                lr_counter += 1

            e_loss, e_r, e_p, e_f1 = 0, 0, 0, 0  # epoch

            for i, batch in enumerate(train_data):
                data, labels = batch[0], batch[1].astype('float32')

                # data = data.as_in_context(context=self.ctx)
                # labels = labels.as_in_context(context=self.ctx)
                data = split_and_load(data, ctx_list=self.ctx, batch_axis=0, even_split=False)
                labels = split_and_load(labels, ctx_list=self.ctx, batch_axis=0, even_split=False)

                # self.print_info('data: {}, labels: {}'.format(data.shape, labels.shape))

                with autograd.record():  # 梯度求导
                    # outputs = base_net(data)
                    # bc_loss = loss_func(outputs, labels)
                    outputs = [base_net(X) for X in data]
                    bc_loss = [loss_func(yhat, y) for yhat, y in zip(outputs, labels)]

                # autograd.backward(bc_loss)
                for l in bc_loss:
                    l.backward()

                trainer.step(self.batch_size)

                # self.print_info('Loss: {}'.format(bc_loss.shape))  # (8, 27)
                batch_loss = sum([l.mean().asscalar() for l in bc_loss]) / len(bc_loss)  # batch的loss
                e_loss += batch_loss

                br, bp, bf1 = 0, 0, 0
                for output, label in zip(outputs, labels):
                    r, p, f1 = self.metric_of_rpf(output, label)
                    br += r
                    bp += p
                    bf1 += f1

                e_r += br
                e_p += bp
                e_f1 += bf1

                sp_batch = len(outputs)
                self.print_info('batch: {}, loss: {:.5f}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                                .format(i, batch_loss, br / sp_batch, bp / sp_batch, bf1 / sp_batch))

            n_batch = safe_div(len(train_data), self.batch_size)
            e_loss /= n_batch
            e_r /= n_batch
            e_p /= n_batch
            e_f1 /= n_batch

            self.print_info('epoch: {}, loss: {:.5f}, recall: {:.2f}, precision: {:.2f}, f1: {:.2f}'
                            .format(epoch, e_loss, e_r, e_p, e_f1))
            self.val_net(base_net, val_data)


if __name__ == '__main__':
    mlt = MultiLabelTrainer()
    mlt.train_model()
