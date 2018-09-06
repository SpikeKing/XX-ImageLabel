#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/8/21
"""
from itertools import chain

from utils.project_utils import unicode_list

CONTENT_TAGS = {
    u'护肤': [
        u'洁面', u'精华', u'爽肤水', u'喷雾', u'眼霜',
        u'颈霜', u'面膜', u'乳液面霜', u'防晒', u'美容仪器',
        u'效果'
    ],
    u'美妆': [
        u'遮瑕', u'粉饼', u'粉底', u'眉毛', u'眼影',
        u'眼线', u'妆容', u'睫毛', u'腮红', u'修容',
        u'高光', u'口红', u'美瞳', u'隔离', u'香水',
        u'试色', u'化妆工具'
    ],
    u'美甲': [u'美甲工具', u'美甲推荐', u'美甲教程', u'指甲油'],
    u'美发': [
        u'发型', u'美发教程', u'发色', u'护发', u'美发工具',
        u'饰品'
    ],
    u'美体': [u'沐浴露', u'爽身粉', u'身体乳', u'美体效果', u'磨砂膏', u'美体工具'],
    u'穿搭': [
        u'穿搭教程', u'潮牌', u'鞋', u'包', u'表',
        u'手链', u'眼镜', u'资讯', u'帽子', u'街拍',
        u'项链', u'戒指', u'耳饰', u'其他穿搭'],
    u'探店': [
        u'饮品店', u'书店', u'甜品店', u'酒店', u'服装店',
        u'鞋店', u'民宿', u'餐厅', u'商场', u'展览'
    ],
    u'拍照修图': [u'修图工具', u'修图教程', u'拍照工具', u'拍照教程', u'摄影作品'],
    u'美食': [u'教程', u'记录'],
    u'旅游': [u'攻略', u'购物'],
    u'健身': [u'瘦全身', u'瘦局部', u'身体塑性', u'运动'],
    u'音乐': [u'唱歌', u'乐器', u'乐谱', u'音乐会', u'音乐推荐'],
    u'结婚': [u'婚纱', u'婚照', u'备婚', u'婚礼'],
    u'生活家居': [u'生活物品', u'家居物品', u'生活技巧', u'装修'],
    u'数码': [
        u'手机', u'相机', u'游戏机', u'电脑', u'键盘',
        u'音响', u'小程序', u'配件', u'耳机', u'平板',
        u'手表', u'APP', u'其他数码'
    ],
    u'游戏': [u'游戏技巧', u'游戏推荐'],
    u'情感': [u'职场', u'校园', u'生活', u'爱情'],
    u'宠物': [u'宠物用品', u'养宠常识', u'宠物互动', u'宠物日常'],
    u'书籍': [u'书籍推荐', u'原创故事'],
    u'二次元': [u'cosplay', u'古风汉服', u'动漫推荐', u'手办周边', u'二次元常识'],
    u'影视综艺': [u'电影推荐', u'电视剧推荐', u'综艺推荐', u'预告资讯', u'影评解读', u'杂谈'],
    u'美容': [],
    u'手工': [],
    u'玩具': [],
    u'文物': [],
    u'艺术': [u'建筑设计', u'书法', u'画画'],
    u'明星': [u'明星发布', u'粉丝追星', u'八卦资讯'],
}

SUB_CONTENT_TAGS = {
    u'效果': [
        u'补水', u'角质', u'美白', u'修复', u'祛斑',
        u'毛孔', u'抗衰', u'祛痘'
    ],
    u'眼影': [u'眼影推荐', u'眼影试色'],
    u'口红': [u'口红推荐', u'口红试色'],
    u'化妆工具': [
        u'假睫毛', u'双眼皮贴', u'化妆刷', u'粉扑', u'睫毛夹',
        u'其他'
    ],
    u'美体效果': [u'身体美白', u'脱毛', u'妊娠纹'],
    u'饮品店': [u'奶茶', u'果蔬饮品', u'茶', u'咖啡', u'冰品', u'糖水', u'酒'],
    u'教程': [u'中餐教程', u'西餐教程', u'烘焙教程', u'饮品教程'],
    u'记录': [
        u'东南亚菜', u'火锅', u'甜品', u'海鲜', u'水果',
        u'小吃零食', u'印度菜', u'西餐', u'韩餐', u'日料',
        u'饮品', u'中餐', u'烧烤', u'黑暗料理'
    ],
    u'攻略': [u'旅行路线', u'旅行常识'],
    u'瘦局部': [u'瘦胳膊', u'瘦腰腹', u'瘦腿', u'瘦脸'],
    u'身体塑性': [u'肩颈', u'背部', u'臂部', u'拉伸'],
    u'运动': [u'游泳', u'极限运动', u'瑜伽', u'滑板', u'潜水'],
    u'生活物品': [u'好物安利', u'新品推荐'],
}


def traverse_tags():
    """
    遍历全部标签
    :return: 标签列表
    """
    one_level = unicode_list(list(CONTENT_TAGS.keys()))
    two_level = unicode_list(list(chain.from_iterable(CONTENT_TAGS.values())))

    ot_error = set(one_level) & set(two_level)
    if ot_error:
        raise Exception('一级和二级标签重复!')

    one_two_level = one_level + two_level
    three_level = unicode_list(list(chain.from_iterable(SUB_CONTENT_TAGS.values())))
    oth_error = set(one_two_level) & set(three_level)
    if oth_error:
        raise Exception('三级标签重复! %s' % oth_error)

    all_tags = one_two_level + three_level
    print('标签数量: {}'.format(len(all_tags)))

    return unicode_list(all_tags)


def main():
    # all_tags = traverse_tags()
    # print(len(all_tags))
    print(len(CONTENT_TAGS.keys()))


if __name__ == '__main__':
    main()
