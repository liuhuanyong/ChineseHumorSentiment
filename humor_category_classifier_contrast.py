#!/usr/bin/env python3
# coding: utf-8
# File: humor_category_classify.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-11-10

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.layers.core import *
import matplotlib.pyplot as plt
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class HumorCategoryClassify:
    def __init__(self):
        self.cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(self.cur, 'data/humor_category_train.txt')
        self.vocab_path = os.path.join(self.cur, 'model/vocab_humor_category.txt')
        self.embedding_file = os.path.join(self.cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(self.cur, 'tokenvec_bilstm2_humor_category_cnn_att.h5')
        self.datas, self.word_dict = self.build_data()
        self.class_dict = {
            0: "谐音幽默",
            1: "谐义幽默",
            2: "反转幽默",
        }
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 100
        self.BATCH_SIZE = 256
        self.LIMIT_RATE = 0.95
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = self.select_best_length()
        self.embedding_matrix = self.build_embedding_matrix()
        self.SINGLE_ATTENTION_VECTOR = False

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
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
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
            sum_length += i[1] * i[0]
        average_length = sum_length / all_sent
        for i in len_dict:
            rate = i[1] / all_sent
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
            cate = int(data[1]) - 1
            label_vector = self.label_onehot(cate)
            y_train.append(label_vector)

        x_train = pad_sequences(x_train, self.TIME_STAMPS)
        return np.array(x_train), np.array(y_train)

    '''对数据进行onehot映射操作'''
    def label_onehot(self, label):
        one_hot = [0] * self.NUM_CLASSES
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

    '''多个CNN作为特征提取器'''
    def model_cnn(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=False)
        origin_inputs = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        embedding_inputs = embedding_layer(origin_inputs)
        # 1-gram信息
        conv1 = Conv1D(128, 1)(embedding_inputs)
        pool1 = MaxPooling1D(self.TIME_STAMPS - 1 + 1)(conv1)
        pool1 = Flatten()(pool1)
        # 2-gram信息
        conv2 = Conv1D(128, 2)(embedding_inputs)
        pool2 = MaxPooling1D(self.TIME_STAMPS - 2 + 1)(conv2)
        pool2 = Flatten()(pool2)
        # 3-gram信息
        conv3 = Conv1D(128, 3)(embedding_inputs)
        pool3 = MaxPooling1D(self.TIME_STAMPS - 3 + 1)(conv3)
        pool3 = Flatten()(pool3)
        # 4-gram信息
        conv4 = Conv1D(128, 4)(embedding_inputs)
        pool4 = MaxPooling1D(self.TIME_STAMPS - 4 + 1)(conv4)
        pool4 = Flatten()(pool4)
        # 5-gram信息
        conv5 = Conv1D(128, 5)(embedding_inputs)
        pool5 = MaxPooling1D(self.TIME_STAMPS - 5 + 1)(conv5)
        pool5 = Flatten()(pool5)
        # 6-gram信息
        conv6 = Conv1D(128, 6)(embedding_inputs)
        pool6 = MaxPooling1D(self.TIME_STAMPS - 6 + 1)(conv6)
        pool6 = Flatten()(pool6)
        # n-gram信息拼接
        merged = merge.concatenate([pool1, pool2, pool2, pool3, pool4, pool5, pool6],axis=1)
        merged = BatchNormalization()(merged)
        output = Dense(self.NUM_CLASSES, activation='softmax')(merged)
        model = Model(input=[origin_inputs], output=output)
        print(model.summary())
        return model

    '''增加attention机制的cnn模型'''
    def model_cnn_attention(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=False)
        origin_inputs = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        embedding_inputs = embedding_layer(origin_inputs)
        # 1-gram信息
        conv1 = Conv1D(128, 1)(embedding_inputs)
        pool1 = MaxPooling1D(self.TIME_STAMPS - 1 + 1)(conv1)
        pool1 = Flatten()(pool1)
        # 2-gram信息
        conv2 = Conv1D(128, 2)(embedding_inputs)
        pool2 = MaxPooling1D(self.TIME_STAMPS - 2 + 1)(conv2)
        pool2 = Flatten()(pool2)
        # 3-gram信息
        conv3 = Conv1D(128, 3)(embedding_inputs)
        pool3 = MaxPooling1D(self.TIME_STAMPS - 3 + 1)(conv3)
        pool3 = Flatten()(pool3)
        # 4-gram信息
        conv4 = Conv1D(128, 4)(embedding_inputs)
        pool4 = MaxPooling1D(self.TIME_STAMPS - 4 + 1)(conv4)
        pool4 = Flatten()(pool4)
        # 5-gram信息
        conv5 = Conv1D(128, 5)(embedding_inputs)
        pool5 = MaxPooling1D(self.TIME_STAMPS - 5 + 1)(conv5)
        pool5 = Flatten()(pool5)
        # 6-gram信息
        conv6 = Conv1D(128, 6)(embedding_inputs)
        pool6 = MaxPooling1D(self.TIME_STAMPS - 6 + 1)(conv6)
        pool6 = Flatten()(pool6)
        # n-gram信息拼接
        merged = merge.concatenate([pool1, pool2, pool2, pool3, pool4, pool5, pool6], axis=1)
        merged = BatchNormalization()(merged)
        # 池化之后,加入attention
        merged_output_dim = int(merged.shape[1])
        attention_probs = Dense(merged_output_dim, activation='softmax', name='attention_vec')(merged)
        attention_mul = merge.multiply([merged, attention_probs], name='attention_mul')
        # Attention之后,接Dense全连接层
        attention_mul = Dense(64)(attention_mul)
        #Dense全连接层后面再接一个全连接,加入softmax进行分类
        output = Dense(self.NUM_CLASSES, activation='softmax')(attention_mul)
        model = Model(input=[origin_inputs], output=output)
        return model

    '''attention层计算'''
    def attention_3d_block(self, inputs):
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, self.TIME_STAMPS))(a)
        a = Dense(self.TIME_STAMPS, activation='softmax')(a)
        if self.SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = merge.multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    '''在LSTM层之后加入attention + BatchNormalization'''
    def model_bilstm_attention_after(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=False)
        origin_inputs = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        embedding_inputs = embedding_layer(origin_inputs)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding_inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        attention_mul = self.attention_3d_block(lstm_out)
        attention_mul = Flatten()(attention_mul)
        output = Dense(self.NUM_CLASSES, activation='softmax')(attention_mul)
        model = Model(input=[origin_inputs], output=output)
        return model

    '''在lstm层之前加入attention'''
    def model_bilstm_attention_before(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=False)
        origin_inputs = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        embedding_inputs = embedding_layer(origin_inputs)
        attention_mul = self.attention_3d_block(embedding_inputs)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(attention_mul)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Flatten()(lstm_out)
        output = Dense(self.NUM_CLASSES, activation='softmax')(lstm_out)
        model = Model(input=[origin_inputs], output=output)
        return model

    '''在lstm层之前加入attention'''
    def model_bilstm(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=False)
        origin_inputs = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        embedding_inputs = embedding_layer(origin_inputs)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding_inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Flatten()(lstm_out)
        output = Dense(self.NUM_CLASSES, activation='softmax')(lstm_out)
        model = Model(input=[origin_inputs], output=output)
        return model

    '''训练模型函数式'''
    def train_model(self, model_name='bilstm'):
        x_train, y_train = self.modify_data()
        model = self.model_bilstm()
        if model_name == 'cnn':
            model = self.model_cnn()
        elif model_name == 'cnn_attention':
            model = self.model_cnn_attention()
        elif model_name == 'bilstm':
            model = self.model_bilstm()
        elif model_name == 'bilstm_attention_before':
            model = self.model_bilstm_attention_before()
        elif model_name == 'bilstm_attention_after':
            model = self.model_bilstm_attention_after()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        history = model.fit([x_train],
                            y_train,
                            validation_split=0.2,
                            batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS)
        self.draw_train(history, model_name)
        model_path = os.path.join(self.cur, 'model/%s.model'%model_name)
        model.save(model_path)
        return model

    '''绘制训练曲线'''
    def draw_train(self, history, model_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy- %s'%model_name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # 保存训练准确度图片
        image_filepath = os.path.join(self.cur, 'train_image/%s_accu.png'%model_name)
        plt.savefig(image_filepath)
        # # Plot training & validation loss values
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss - %s'%model_name)
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # # 保存训练loss图片
        # image_filepath = os.path.join(self.cur, 'train_image/%s_loss.png'%model_name)
        # plt.savefig(image_filepath)
        plt.close()

    '''对比测试'''
    def test_model(self):
        self.train_model('cnn')
        self.train_model('cnn_attention')
        self.train_model('bilstm')
        self.train_model('bilstm_attention_before')
        self.train_model('bilstm_attention_after')


if __name__ == '__main__':
    handler = HumorCategoryClassify()
    handler.test_model()
