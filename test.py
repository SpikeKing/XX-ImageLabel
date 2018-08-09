#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/31
"""

from PIL import Image, ImageDraw


def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    image.save('kkk.jpg')
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # new_image.show()
    new_image.save('aaa.jpg')

    return new_image


def main():
    img_path = '/Users/wang/workspace/XX-ImageLabel/img_data/logAll/NvXing_145.jpg'
    image = Image.open(img_path)
    size = (416, 416)
    pad_image(image, size)  # 填充图像


if __name__ == '__main__':
    main()
