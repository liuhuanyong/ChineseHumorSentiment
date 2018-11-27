#!/usr/bin/env python3
# coding: utf-8
# File: spider.py.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-9-13

from urllib import request
from lxml import etree
import pymongo
import chardet

class HumorSpider:
    def __init__(self):
        self.conn = pymongo.MongoClient()
        self.db = self.conn['humor']
        self.col = self.db['news']

    '''获取原网页'''
    def get_html(self, url):
        html = request.urlopen(url).read()
        encoding = chardet.detect(html)['encoding']
        return html.decode(encoding, 'ignore')

    '''juben68采集主函数'''
    def juben68_spider(self):
        count = 0
        links = ['http://www.juben68.com/index_%s.html'%i for i in range(1, 16)]
        for link in links:
            html = self.get_html(link)
            selector = etree.HTML(html)
            urls = ['http://www.juben68.com'+ i for i in selector.xpath('//div[@class="block"]/h2/a/@href')]
            for url in urls:
                count += 1
                print(count, url)
                if count < 1262:
                    continue
                data = {}
                data['url'] = url
                body = []
                news = self.get_html(url)
                selector = etree.HTML(news)
                try:
                    ps = [selector.xpath('//div[@class="post_content"]/p')][0]
                    for p in ps:
                        content = p.xpath('string(.)').replace('\r\n','').replace('\t', '').replace('\xa0', '').replace(' ','').replace('：',':')
                        if not content:
                            continue
                        if '后面更精彩' in content:
                            break
                        body.append(content)
                    data['content'] = '\n'.join(body)
                    self.col.insert(data)
                except Exception as e:
                    print(e)
                    pass

    '''xsxpw采集主函数'''
    def xsxpw_spider(self):
        count = 0
        links = ['http://www.xsxpw.com/juben/xiaopinjuben/index_%s.html'%i for i in range(2, 153)]
        links.append('http://www.xsxpw.com/juben/xiaopinjuben/')
        for link in links:
            html = self.get_html(link)
            selector = etree.HTML(html)
            urls = ['http://www.xsxpw.com'+ i for i in selector.xpath('//li[@class="list_title 3"]/a/@href')]
            for url in urls:
                count += 1
                print(count, url)
                data = {}
                data['url'] = url
                news = self.get_html(url)
                try:
                    title = selector.xpath('//title/text()')[0]
                    content = news.split('<div class="newsbody">')[1].split('<div class="page-nav">')[0].split('</')[0].replace('br', 'BR').replace('\u3000', '').replace('\x00','').replace('\r', '').replace('\n', '')
                    body = content.split('<BR>')
                    data['content'] = '\n'.join(body)
                    data['title'] = title
                    self.col.insert(data)
                except Exception as e:
                    print(e)

if __name__=='__main__':
    handler = HumorSpider()
    handler.xsxpw_spider()
    handler.juben68_spider()