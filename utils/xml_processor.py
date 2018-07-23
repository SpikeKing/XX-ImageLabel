#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/20
"""
import os
import xml.etree.ElementTree as ET
import xmltodict
import xml.dom.minidom as minidom

from root_dir import ROOT_DIR
from utils.project_utils import read_file


def read_anno_xml(xml_file):
    """
    读取XML的文件
    :param xml_file: XML的文件
    :return:
    """
    xml_dict = xmltodict.parse(read_file(xml_file, mode='one'))
    boxes_list = []

    try:
        annotation = xml_dict['annotation']
        if 'object' in annotation:
            objects = annotation['object']
            if isinstance(objects, list):
                for object in objects:
                    bndbox = object['bndbox']
                    boxes_list.append([bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']])
            else:
                bndbox = objects['bndbox']
                boxes_list.append([bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']])
    except Exception as e:
        pass

    return boxes_list


if __name__ == '__main__':
    fp = os.path.join(ROOT_DIR, "img_data/logAll-0717/ErTong_6.xml")
    boxes_list = read_anno_xml(fp)
    print(boxes_list)
