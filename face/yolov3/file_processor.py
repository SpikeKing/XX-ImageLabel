#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/16
"""
import json
import os

from PIL import Image

from face.yolov3.yolov3_dir import OUTPUT_DATA
from root_dir import IMG_DATA
from utils.project_utils import read_file, write_line


def reorder_boxes():
    in_file = os.path.join(OUTPUT_DATA, 'logAll_out.txt')
    out_file = os.path.join(OUTPUT_DATA, 'logAll_json.txt')

    data_lines = read_file(in_file)

    for data_line in data_lines:
        items = data_line.split(' ')
        img_name = items[0]
        json_dict = dict()
        json_dict['tag'] = 'face'
        json_dict['name'] = img_name.split('/')[-1]
        image = Image.open(os.path.join(IMG_DATA, img_name))

        if 'size' not in json_dict:
            json_dict['size'] = dict()
        json_dict['size']['width'] = image.width
        json_dict['size']['height'] = image.height

        if 'label' not in json_dict:
            json_dict['label'] = []

        if items >= 2:
            boxes = items[1:]
            for box in boxes:
                box_dict = dict()
                x_min, y_min, x_max, y_max, label = box.split(',')
                box_dict['minX'] = x_min
                box_dict['minY'] = y_min
                box_dict['maxX'] = x_max
                box_dict['maxY'] = y_max
                box_dict['tag'] = 'face'
                box_dict['tagId'] = 0
                json_dict['label'].append(box_dict)

        json_str = json.dumps(json_dict)
        print(json_str)
        write_line(out_file, json_str)


if __name__ == '__main__':
    reorder_boxes()
