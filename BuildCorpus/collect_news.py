#!/usr/bin/env python3
# coding: utf-8
# File: collect_news.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-9-17

import pymongo
class CollectCorpus:
    def __init__(self):
        self.conn = pymongo.MongoClient()
        self.db = self.conn['humor']
        self.col = self.db['news']
        self.col2 = self.db['news2']
    
    '''收集dialog语料'''
    def collect_dialog(self):
        count = 0
        sum = 0
        for item in self.col.find():
            url = item['url']
            content = item['content'].replace('：',':')
            lines = content.split('\n')
            dialogs = []
            if url:
                count += 1
                for line in lines:
                    if self.check_nosiy(line) and len([i for i in line.split(':') if i]) > 1 and len(line.split(':')[0]) < 5:
                        dialogs.append(line)
            if len(dialogs) > 2:
                sum += len(dialogs)
                f = open('dialog/%s.txt'%count, 'w+')
                f.write('\n'.join(dialogs))
                f.close()

        for item in self.col2.find():
            url = item['url']
            content = item['content'].replace('：',':').replace('&nbsp;','').replace(' ','').replace('http://www.juben68.com','')
            lines = content.split('\n')
            dialogs = []
            if url:
                count += 1
                for line in lines:
                    if self.check_nosiy(line) and len([i for i in line.split(':') if i]) > 1 and len(line.split(':')[0]) < 5:
                        dialogs.append(line)
            if len(dialogs) > 3:
                sum += len(dialogs)
                f = open('dialog/%s.txt'%count, 'w+')
                f.write('\n'.join(dialogs))
                f.close()
        print(sum)
        return

    '''移除噪声'''
    def check_nosiy(self, s):
        nosiy = ['小品', '编剧','作者', '人物', '场景', '地点', '创作', '表演','地点','时间','旁白','道具','背景']
        speaker =  s.split(':')[0]
        for wd in nosiy:
            if wd in speaker:
                return 0
        return 1



handler = CollectCorpus()
handler.collect_dialog()
