#!/usr/bin/env python3
# coding: utf-8
# File: yinyu_senti_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class YinyuSentimentClassify:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/yinyu_senti_train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab_yinyu_senti.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_yinyu_senti.h5')
        self.datas, self.word_dict = self.build_data()
        self.class_dict = {
            0:"乐",
            1:"好",
            2:"怒",
            3: "哀",
            4: "惧",
            5: "恶",
            6: '惊',
        }
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
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
        2904/2904 [==============================] - 88s 30ms/step - loss: 1.5227 - acc: 0.4690 - val_loss: 1.4170 - val_acc: 0.4545
        2904/2904 [==============================] - 73s 25ms/step - loss: 1.2590 - acc: 0.5606 - val_loss: 1.4097 - val_acc: 0.5041
        2904/2904 [==============================] - 72s 25ms/step - loss: 1.0834 - acc: 0.6250 - val_loss: 1.3585 - val_acc: 0.4917
        2904/2904 [==============================] - 72s 25ms/step - loss: 0.9141 - acc: 0.6980 - val_loss: 1.3292 - val_acc: 0.5358
        2904/2904 [==============================] - 72s 25ms/step - loss: 0.7456 - acc: 0.7521 - val_loss: 1.5308 - val_acc: 0.4848
        2904/2904 [==============================] - 71s 25ms/step - loss: 0.5959 - acc: 0.8134 - val_loss: 1.5379 - val_acc: 0.5399
        2904/2904 [==============================] - 72s 25ms/step - loss: 0.4306 - acc: 0.8650 - val_loss: 1.6575 - val_acc: 0.5152
        2904/2904 [==============================] - 69s 24ms/step - loss: 0.3548 - acc: 0.9025 - val_loss: 1.8639 - val_acc: 0.5303
        2904/2904 [==============================] - 67s 23ms/step - loss: 0.2896 - acc: 0.9174 - val_loss: 1.8765 - val_acc: 0.4931
        2904/2904 [==============================] - 81s 28ms/step - loss: 0.1923 - acc: 0.9446 - val_loss: 2.1730 - val_acc: 0.4931
        '''
if __name__ == '__main__':
    ner = YinyuSentimentClassify()
    ner.train_model()
