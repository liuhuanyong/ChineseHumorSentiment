#!/usr/bin/env python3
# coding: utf-8
# File: lstm_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class HumorDegreeClassify:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/humor_degree_train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab_hummor_degree.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_humor_degree.h5')
        self.datas, self.word_dict = self.build_data()
        self.class_dict ={
                         '一级': 1,
                         '二级': 2,
                         '三级': 3,
                         '四级': 4,
                         '五级': 5,
                        }
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 256
        self.LIMIT_RATE = 0.95
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = self.select_best_length()
        self.embedding_matrix = self.build_embedding_matrix()

    '''构造数据集'''
    def build_data(self):
        datas = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = line.rstrip().split('\t')
            if len(line) != 2:
                continue
            sent = line[0]
            cate = line[1]
            wds = [char for char in sent]
            for wd in wds:
                vocabs.add(wd)
            datas.append([wds, cate])
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = line.strip().split('	')
            if not line:
                continue
            sent = line[0]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)
        return max_length

    '''将数据转换成keras所需的格式'''
    def modify_data(self):
        x_train = []
        for data in self.datas:
            x_vectors = []
            for wd in data[0]:
                x_vectors.append(self.word_dict.get(wd))
            x_train.append(x_vectors)

        y_train = []
        for data in self.datas:
            cate = int(data[1])-1
            label_vector = self.label_onehot(cate)
            y_train.append(label_vector)

        x_train = pad_sequences(x_train, self.TIME_STAMPS)

        return np.array(x_train), np.array(y_train)

    '''对数据进行onehot映射操作'''
    def label_onehot(self, label):
        one_hot = [0]*self.NUM_CLASSES
        one_hot[int(label)] = 1
        return one_hot

    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    '''使用预训练向量进行模型训练'''
    def tokenvec_bilstm2_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(32), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    '''训练模型'''
    def train_model(self):
        x_train, y_train = self.modify_data()
        model = self.tokenvec_bilstm2_model()
        history = model.fit(x_train,
                            y_train,
                            validation_split= 0.2,
                            batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS)
        self.draw_train(history)
        model.save(self.model_path)
        model = 1
        return model

    '''绘制训练曲线'''
    def draw_train(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        # 7836/7836 [==============================] - 205s 26ms/step - loss: 17.1782 - acc: 0.9624
        '''
        6436/6436 [==============================] - 450s 70ms/step - loss: 0.7774 - acc: 0.5489 - val_loss: 0.7649 - val_acc: 0.5081
        6436/6436 [==============================] - 413s 64ms/step - loss: 0.6728 - acc: 0.6019 - val_loss: 0.6609 - val_acc: 0.6062
        6436/6436 [==============================] - 392s 61ms/step - loss: 0.6152 - acc: 0.6641 - val_loss: 0.6620 - val_acc: 0.6112
        6436/6436 [==============================] - 403s 63ms/step - loss: 0.5456 - acc: 0.7253 - val_loss: 0.6672 - val_acc: 0.6130
        6436/6436 [==============================] - 397s 62ms/step - loss: 0.4518 - acc: 0.7935 - val_loss: 0.7853 - val_acc: 0.6050
        6436/6436 [==============================] - 393s 61ms/step - loss: 0.3636 - acc: 0.8442 - val_loss: 0.9156 - val_acc: 0.6043
        6436/6436 [==============================] - 392s 61ms/step - loss: 0.2716 - acc: 0.8891 - val_loss: 0.9651 - val_acc: 0.6137
        6436/6436 [==============================] - 393s 61ms/step - loss: 0.2114 - acc: 0.9159 - val_loss: 1.0418 - val_acc: 0.5845
        6436/6436 [==============================] - 403s 63ms/step - loss: 0.1534 - acc: 0.9447 - val_loss: 1.3358 - val_acc: 0.5826
        6436/6436 [==============================] - 844s 131ms/step - loss: 0.1309 - acc: 0.9504 - val_loss: 1.3084 - val_acc: 0.6075
        6436/6436 [==============================] - 1121s 174ms/step - loss: 0.1018 - acc: 0.9638 - val_loss: 1.5146 - val_acc: 0.5901
        6436/6436 [==============================] - 898s 140ms/step - loss: 0.0825 - acc: 0.9699 - val_loss: 1.5126 - val_acc: 0.6000
        6436/6436 [==============================] - 432s 67ms/step - loss: 0.0743 - acc: 0.9744 - val_loss: 1.4816 - val_acc: 0.6031
        6436/6436 [==============================] - 374s 58ms/step - loss: 0.0677 - acc: 0.9770 - val_loss: 1.6247 - val_acc: 0.5994
        6436/6436 [==============================] - 375s 58ms/step - loss: 0.0597 - acc: 0.9803 - val_loss: 1.8652 - val_acc: 0.5901
        6436/6436 [==============================] - 376s 58ms/step - loss: 0.0455 - acc: 0.9849 - val_loss: 1.9099 - val_acc: 0.6025
        6436/6436 [==============================] - 594s 92ms/step - loss: 0.0476 - acc: 0.9828 - val_loss: 1.8380 - val_acc: 0.6056
        6436/6436 [==============================] - 797s 124ms/step - loss: 0.0395 - acc: 0.9868 - val_loss: 1.9119 - val_acc: 0.5957
        6436/6436 [==============================] - 832s 129ms/step - loss: 0.0379 - acc: 0.9879 - val_loss: 1.9391 - val_acc: 0.5957
        6436/6436 [==============================] - 797s 124ms/step - loss: 0.0402 - acc: 0.9854 - val_loss: 2.1138 - val_acc: 0.6012
        '''

if __name__ == '__main__':
    ner = HumorDegreeClassify()
    ner.train_model()