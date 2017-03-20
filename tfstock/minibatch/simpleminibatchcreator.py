# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from .baseminibatchcreator import *

class SimpleMinibatchCreator(BaseMinibatchCreator):
    u"""保持しているdfからランダムなデータセットを返却するミニバッチ生成器"""

    def __init__(self, df, parameter):
        u"""1:9の割合で、テスト用と訓練用に分割します"""

        self.parameter = parameter

        # dfの生成
        index = df.index
        columns = df.columns
        self.df_train = pd.DataFrame(index=index, columns=columns)
        self.df_test = pd.DataFrame(index=index, columns=columns)

        # 1:9の割合で分割して格納
        for termid, subdf in df.groupby(level=0):
            if np.random.rand() > 0.1:
                self.df_train.concat(subdf)
            else:
                self.df_test.concat(subdf)

    def get_next_batch_train(self, batch_size):
        u"""訓練用のデータセットをbatch_size分返却します"""
        return self.get_next_batch(batch_size, self.df_train)

    def get_next_batch_test(self, batch_size):
        u"""テストのデータセットをbatch_size分返却します"""
        return self.get_next_batch(batch_size, self.df_test)

    def get_next_batch(self, batch_size, df):
        u"""batch_size で指定した数のデータセットを返却する
         データセットの形式
         x... 1 x startdate_index
         y... [0,0,0,0,1,0,...] one hot
         minibatch としては、ラベルが均等な数現れるように返却する
         """

        batch_x = []
        batch_y = []

        for ibatch in range(batch_size):

            # 返却するデータセットのラベルをランダムに決定
            label = int(np.random.rand() * self.parameter.num_of_class)

            # 渡されたdfから、ラベルが label のセットをランダムに選択

            # inputの配列を作成
            # normalizedのみのため要素数は1
            list_input = []
            # column についてのループ
            # for column in df.column
            #   indexとlabel列以外を使用する
            #   list_input.append(df)[0:parameter.startdate_index])
            # TODO

            # 行列の入れ替えをして返却するリストに追加
            list_input_reshaped = zip(*list_input)
            batch_x.append(list_input_reshaped)

            # label
            list_label = []
            # TODO
            batch_y.append(list_label)

        return batch_x, batch_y


class SimpleMinibatchCreatorParameter:
    u"""SimpleMinibatchCreatorで使用するパラメタを定義します"""
    def __init__(self, parameter):
        self.num_of_class = parameter.num_of_class
        self.statedate_index = parameter.startdate_index

