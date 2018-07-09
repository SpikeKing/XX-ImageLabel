#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/9
"""
import argparse
import os
from multiprocessing.pool import ThreadPool

import requests
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR
from utils.log_utils import print_info
from utils.project_utils import *


class ImgDownloader(object):
    def __init__(self, out_folder=None):
        """
        构造器
        :param out_folder: 图片存储文件夹
        """
        self.out_folder = out_folder
        self.imgs_names = None  # 用于检测重复图片
        pass

    def download_img(self, img_url):
        """
        下载图片
        :param img_url: 图片URL
        :return: None
        """
        img_name = img_url.split('/')[-1]  # 图片文件名

        if not self.imgs_names:
            _, self.imgs_names = traverse_dir_files(self.out_folder)

        if img_name in self.imgs_names:
            print_info('图片已存在: %s' % img_name)
            return

        img_data = requests.get(img_url).content

        out_file = os.path.join(self.out_folder, img_name)  # 输出文件

        with open(out_file, 'wb') as hl:
            hl.write(img_data)
            print_info('图片已下载: %s' % img_name)

    def download_imgs(self, img_file, out_folder):
        """
        下载图片集合
        :param img_file: 图片文件
        :param out_folder: 文件夹
        :return: None
        """
        paths_list = read_file(img_file)
        print_info('图片总数: %s' % len(paths_list))
        count = 0
        self.out_folder = out_folder  # 存储文件夹
        for path in paths_list:
            self.download_img(path)
            count += 1
            if count % 200 == 0:
                print_info('已下载: %s' % count)

    def download_imgs_for_mp(self, img_file, out_folder, n_thread=10):
        """
        多线程下载
        :param img_file: 图片文件
        :param out_folder: 输出文件夹
        :param n_thread: 线程数, 默认10个
        :return: None
        """
        print_info('线程总数：%s' % n_thread)
        pool = ThreadPool(n_thread)  # 多线程下载
        paths_list = read_file(img_file)
        self.out_folder = out_folder  # 存储文件夹
        pool.map(self.download_img, paths_list)
        pool.close()
        pool.join()
        print_info('全部下载完成')


def test_of_ImgDownloader():
    """
    测试
    """
    img_file = os.path.join(ROOT_DIR, 'img_downloader', 'urls', 'log明星')
    out_folder = os.path.join(ROOT_DIR, 'img_data', 'log明星')
    mkdir_if_not_exist(out_folder)  # 新建文件夹
    ider = ImgDownloader()
    ider.download_imgs(img_file, out_folder)


def parse_args():
    """
    处理脚本参数，支持相对路径
    img_file 文件路径，默认文件夹：img_downloader/urls
    out_folder 输出文件夹，默认文件夹：img_data
    :return: arg_img，文件路径；out_folder，输出文件夹
    """
    parser = argparse.ArgumentParser(description='下载数据脚本')
    parser.add_argument('--img_file', required=True, help='文件路径', type=str)
    parser.add_argument('--out_folder', help='输出文件夹', type=str)
    args = parser.parse_args()

    arg_img = args.img_file
    if len(arg_img.split('/')) == 1:
        arg_img = os.path.join(ROOT_DIR, 'img_downloader', 'urls', arg_img)
    print_info("文件路径：%s" % arg_img)

    arg_out = args.out_folder
    if not arg_out:
        file_name = arg_img.split('/')[-1]
        arg_out = os.path.join(ROOT_DIR, 'img_data', file_name)
    elif len(arg_out.split('/')) == 1:
        arg_out = os.path.join(ROOT_DIR, 'img_data', arg_out)
    print_info("输出文件夹：%s" % arg_out)
    return arg_img, arg_out


def main():
    """
    入口函数
    """
    arg_img, arg_out = parse_args()
    mkdir_if_not_exist(arg_out)  # 新建文件夹
    ider = ImgDownloader()
    ider.download_imgs_for_mp(arg_img, arg_out)


if __name__ == '__main__':
    main()
