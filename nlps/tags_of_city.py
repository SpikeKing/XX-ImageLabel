#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/9
"""
import os

from nlps.nlp_dir import TXT_DATA
from utils.project_utils import write_line, mkdir_if_not_exist

CHN_CITY_LIST = {
    u'北京': [],
    u'河北': [u'张家口', u'廊坊', u'秦皇岛'],
    u'吉林': [u'延边朝鲜族自治州', u'白山'],
    u'广西': [u'南宁', u'柳州', u'河池', u'桂林'],
    u'江西': [u'上饶', u'南昌', u'景德镇'],
    u'云南': [u'大理', u'香格里拉', u'腾冲', u'丽江', u'昆明'],
    u'湖南': [u'长沙', u'凤凰古城', u'张家界'],
    u'贵州': [u'铜仁', u'贵阳'],
    u'山东': [u'淄博', u'青岛', u'济南'],
    u'上海': [],
    u'宁夏': [],
    u'山西': [u'太原', u'长治', u'忻州'],
    u'天津': [],
    u'重庆': [],
    u'广东': [
        u'汕头', u'肇庆', u'清远', u'佛山', u'江门',
        u'惠州', u'珠海', u'东莞', u'中山', u'潮州',
        u'深圳', u'广州'],
    u'黑龙江': [u'哈尔滨'],
    u'湖北': [u'恩施', u'荆州', u'襄阳', u'十堰', u'武汉'],
    u'浙江': [
        u'温州', u'宁波', u'绍兴', u'湖州', u'余姚',
        u'临海', u'乌镇', u'嘉兴', u'金华', u'舟山',
        u'杭州'],
    u'福建': [
        u'泉州', u'漳州', u'宁德', u'龙岩', u'福州',
        u'厦门'],
    u'内蒙古': [u'呼和浩特', u'呼伦贝尔'],
    u'安徽': [u'合肥'],
    u'江苏': [
        u'无锡', u'苏州', u'扬州', u'镇江', u'常州',
        u'宜兴', u'徐州', u'南京'],
    u'甘肃': [u'兰州', u'甘南藏族自治州', u'张掖', u'武威', u'敦煌'],
    u'台湾': [
        u'基隆', u'台北', u'宜兰', u'新北', u'新竹',
        u'桃园', u'台中', u'台南', u'高雄'],
    u'澳门': [],
    u'香港': [],
    u'西藏': [u'林芝', u'拉萨'],
    u'陕西': [u'渭南', u'延安', u'咸阳', u'西安'],
    u'海南': [u'海口', u'三亚'],
    u'河南': [u'郑州', u'洛阳'],
    u'四川': [u'乐山', u'甘孜藏族自治州', u'绵阳', u'成都'],
    u'青海': [u'西宁', u'海西蒙古族藏族自治州'],
    u'辽宁': [u'东港', u'辽阳', u'沈阳', u'大连'],
    u'新疆': [u'吐鲁番', u'禾木'],
}

WORLD_CITY_LIST = {
    u'美国': [u'夏威夷', u'拉斯维加斯', u'塞班岛', u'洛杉矶', u'旧金山', u'纽约', u'西雅图'],
    u'俄罗斯': [u'摩尔曼斯克', u'符拉迪沃斯托克', u'莫斯科', u'圣彼得堡'],
    u'土耳其': [u'安卡拉', u'孔亚', u'伊斯坦布尔', u'费特希耶'],
    u'柬埔寨': [u'吴哥窑'],
    u'韩国': [u'仁川', u'首尔', u'济州岛'],
    u'越南': [u'芽庄', u'胡志明', u'大叻', u'岘港'],
    u'法国': [u'普罗旺斯', u'法属波利尼西亚', u'尼斯', u'斯特拉斯堡', u'巴黎'],
    u'缅甸': [],
    u'马其他': [u'瓦莱塔'],
    u'墨西哥': [],
    u'日本': [
        u'熊本', u'青森', u'高山', u'境港市', u'滨松',
        u'神户', u'高松', u'伊豆', u'仙台', u'镰仓',
        u'名古屋', u'花卷', u'盛冈', u'九州', u'冲绳岛',
        u'京都', u'大阪', u'长崎', u'奈良', u'北海道',
        u'东京'],
    u'塞舌尔': [],
    u'巴布亚新几内亚': ['阿洛淘'],
    u'丹麦': [],
    u'印尼': [u'巴厘岛'],
    u'英国': [u'曼彻斯特', u'布莱顿', u'苏格兰', u'伦敦'],
    u'泰国': [
        u'清迈', u'象岛', u'斯米兰群岛', u'清莱', u'普吉岛',
        u'芭堤雅', u'苏梅岛', u'曼谷'],
    u'加拿大': [u'蒙特利尔', u'温哥华', u'渥太华', u'多伦多'],
    u'摩洛哥': [u'卡萨布兰卡'],
    u'瑞士': [u'卢塞恩', u'日内瓦'],
    u'捷克': [u'布拉格'],
    u'肯尼亚': [],
    u'冰岛': [u'雷克雅未克', u'阿克雷里'],
    u'新西兰': [u'奥克兰'],
    u'德国': [u'汉堡', u'科隆'],
    u'意大利': [u'佛罗伦萨', u'威尼斯', u'罗马'],
    u'阿联酋': [u'阿布扎比', u'迪拜'],
    u'新加坡': [],
    u'西班牙': [u'塞维利亚', u'马德里', u'巴塞罗那'],
    u'毛里求斯': [],
    u'马来西亚': [u'登嘉楼', u'吉隆坡'],
    u'澳大利亚': [u'悉尼', u'卧龙岗', u'凯恩斯', u'布里斯班', u'墨尔本'],
    u'瑞典': ['斯德哥尔摩'],
    u'伊朗': [],
    u'帕劳': [],
    u'葡萄牙': [u'里斯本'],
    u'菲律宾': [u'帕米拉坎岛', u'宿务', u' 薄荷岛', u'帕斯岛', u'长滩岛'],
    u'埃及': [],
    u'希腊': [u'圣托里尼岛', u'雅典'],
    u'尼泊尔': [],
    u'格鲁吉亚': [],
    u'以色列': [],
    u'纳米比亚': [],
    u'印度': [],
    u'津巴布韦': [],
    u'斯里兰卡': [],
    u'马尔代夫': [],
    u'梵蒂冈': [],
    u'奥地利': [u'维也纳'],
    u'老挝': [u'琅勃拉邦'],
    u'比利时': [],
    u'芬兰': [],
    u'荷兰': [u'阿姆斯特丹']
}


def get_all_cities():
    all_city = list(CHN_CITY_LIST.keys()) + list(WORLD_CITY_LIST.keys())
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
        print('文件已存在!')
        return
    all_city = get_all_cities()

    for city in all_city:
        city_path = os.path.join(kw_path, city)
        write_line(city_path, city)


if __name__ == '__main__':
    print(len(get_all_cities()))
