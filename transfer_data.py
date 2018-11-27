#coding=utf-8
import os
from lxml import etree
from xml.dom.minidom import parse
import xml.dom.minidom


class TransferData:
    def __init__(self):
        return

    def transfer_category(self):
        '''

        谐音幽默(Label=1)、谐义幽默(Label=2)和反转幽默(Label=3)
        '''
        f = open('humor_category_train.txt', 'w+')
        file_name = 'humor_category_train.xml'
        samples = [i.replace('\n','').replace('\t','') for i in open(file_name).read().split('</Humor>')]
        for sample in samples:
            content = sample.split('<Contents>')[-1].split('</Contents>')[0]
            label = sample.split('<Class>')[-1].split('</Class>')[0]
            print(content, label)
            f.write('\t'.join([content, label]) + '\n')
        f.close()

    def transfer_degree(self):
        f = open('humor_degree_train.txt', 'w+')
        file_name = 'humor_degree_train.xml'
        samples = [i.replace('\n', '').replace('\t', '') for i in open(file_name).read().split('</Humor>')]
        for sample in samples:
            content = sample.split('<Contents>')[-1].split('</Contents>')[0]
            label = sample.split('<Level>')[-1].split('</Level>')[0]
            print(content, label)
            f.write('\t'.join([content, label]) + '\n')
        f.close()
        return

    def transfer_senti(self):
        '''
        乐(Label=1)、好(Label=2)、怒(Label=3)、哀(Label=4)、惧(Label=5)、恶(Label=6)、惊(Label=7)
        '''
        f = open('yinyu_senti_train.txt', 'w+')
        file_name = 'yinyu_senti_train.xml'
        samples = [i.replace('\n', '').replace('\t', '') for i in open(file_name).read().split('</metaphor>')]
        for sample in samples:
            content = sample.split('<Sentence>')[-1].split('</Sentence>')[0]
            label = sample.split('<Emo_Class>')[-1].split('</Emo_Class>')[0]
            print(content, label)
            f.write('\t'.join([content, label]) + '\n')
        f.close()
        return

    def transfer_verb(self):
        '''
        动词隐喻(Label=1)、名词隐喻(Label=2)和负例(Label=0, 非隐喻)
        '''
        f = open('yinyu_verb_train.txt', 'w+')
        file_name = 'yinyu_verb_train.xml'
        samples = [i.replace('\n', '').replace('\t', '') for i in open(file_name).read().split('</metaphor>')]
        for sample in samples:
            content = sample.split('<Sentence>')[-1].split('</Sentence>')[0]
            label = sample.split('<Label>')[-1].split('</Label>')[0]
            print(content, label)
            f.write('\t'.join([content, label]) + '\n')
        f.close()
        return


if __name__ == '__main__':
    handler = TransferData()
    handler.transfer_verb()