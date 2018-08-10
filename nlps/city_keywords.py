#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""
import os

from img_downloader.img_comp import mkdir_if_not_exist
from nlps.nlp_dir import TXT_DATA
from utils.project_utils import write_line

CHN_CITY_LIST = {
    u'北京': [],
    u'河北': [u'张家口', u'廊坊', u'秦皇岛'],
    u'吉林': [],
    u'广西': [u'桂林'],
    u'江西': [u'景德镇'],
    u'云南': [u'大理', u'丽江', u'昆明'],
    u'湖南': [u'长沙', u'凤凰古城', u'张家界'],
    u'贵州': [u'贵阳'],
    u'山东': [u'青岛', u'济南'],
    u'上海': [],
    u'天津': [],
    u'重庆': [],
    u'广东': [u'汕头', u'佛山', u'广州', u'深圳'],
    u'黑龙江': [u'哈尔滨'],
    u'湖北': [u'恩施', u'武汉'],
    u'浙江': [u'温州', u'宁波', u'绍兴', u'金华', u'舟山', u'杭州'],
    u'福建': [u'泉州', u'福州', u'厦门'],
    u'安徽': [u'合肥'],
    u'江苏': [u'无锡', u'苏州', u'常州', u'宜兴', u'徐州', u'南京'],
    u'甘肃': [u'兰州', u'武威', u'敦煌'],
    u'台湾': [u'基隆', u'台北', u'台中', u'台南', u'高雄'],
    u'澳门': [],
    u'香港': [],
    u'西藏': [u'拉萨'],
    u'陕西': [u'西安'],
    u'海南': [u'海口', u'三亚'],
    u'河南': [u'郑州', u'洛阳'],
    u'四川': [u'成都'],
    u'青海': [],
    u'辽宁': [u'沈阳', u'大连'],
    u'新疆': [u'禾木'],
}

WORLD_CITY_LIST = {
    u'美国': [u'夏威夷', u'塞班岛', u'洛杉矶', u'旧金山', u'纽约', u'西雅图'],
    u'俄罗斯': [u'摩尔曼斯克', u'莫斯科', u'圣彼得堡'],
    u'土耳其': [],
    u'柬埔寨': [u'吴哥窑'],
    u'韩国': [u'首尔', u'济州岛'],
    u'越南': [u'芽庄', u'大叻', u'岘港'],
    u'法国': [u'斯特拉斯堡', u'巴黎', u'普罗旺斯'],
    u'墨西哥': [],
    u'日本': [u'熊本', u'境港市', u'神户', u'京都', u'大阪', u'新宿', u'长崎', u'奈良', u'北海道', u'东京'],
    u'印尼': [u'巴厘岛'],
    u'英国': [u'布莱顿', u'伦敦'],
    u'泰国': [u'清迈', u'象岛', u'普吉岛', u'芭堤雅', u'苏梅岛', u'曼谷'],
    u'加拿大': [u'蒙特利尔', u'温哥华', u'渥太华', u'多伦多'],
    u'瑞士': [],
    u'捷克': [u'布拉格'],
    u'肯尼亚': [],
    u'冰岛': [u'阿克雷里'],
    u'新西兰': [],
    u'德国': [u'科隆'],
    u'意大利': [u'威尼斯', u'罗马'],
    u'阿联酋': [u'阿布扎比', u'迪拜'],
    u'新加坡': [],
    u'西班牙': [u'塞维利亚', u'马德里', u'巴塞罗那'],
    u'毛里求斯': [],
    u'马来西亚': [],
    u'澳大利亚': [],
    u'帕劳': [],
    u'葡萄牙': [u'里斯本']
}


def get_all_cities():
    all_city = CHN_CITY_LIST.keys() + WORLD_CITY_LIST.keys()
    for cities in CHN_CITY_LIST.values():
        for city in cities:
            all_city.append(city)
    for cities in WORLD_CITY_LIST.values():
        for city in cities:
            all_city.append(city)
    return all_city


def init_city_keywords():
    kw_path = os.path.join(TXT_DATA, 'res_kw', 'cities')
    mkdir_if_not_exist(kw_path)
    if os.path.exists(kw_path):
        print '文件已存在!'
        return
    all_city = get_all_cities()

    for city in all_city:
        city_path = os.path.join(kw_path, city)
        write_line(city_path, city)


if __name__ == '__main__':
    init_city_keywords()
