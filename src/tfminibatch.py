# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tfsummary as tfs

class TFMinibatchHelper:
    
    SUMMARY_FILE_PATH = tfs.TFSummary.SUMMARY_FILE_PATH

    # 
    @classmethod
    def calculateClassLimitsFixed(cls, num_of_class):
        list_class_limit = []
        if num_of_class != 5:
            raise 'FixedのClassLimitsを使う場合、num_of_classは5である必要があります。'
        else:
            list_class_limit = [-2.5, -1.5, 0.5, 1.5, 2.5]
        return list_class_limit

    # サマリデータとクラス数から、全データについてラベルが均等に分布するような
    # クラス分割のしきい値を決定する
    @classmethod
    def calculateClassLimits(cls, num_of_class):

        # サマリデータを取得
        # max, min, value_of_30day, max_after_30day, length
        df_summary = pd.read_csv(cls.SUMMARY_FILE_PATH)

        # ソート
        df_summary = df_summary.sort_values(by='target')

        # target の設定上の最大値と最小値
        target_max = 4.0
        target_min = 1.0

        # 比率が最小値1.0 のデータ数
        n_min_value = len((df_summary.loc[df_summary['target'] == target_min]).index)

        # 合計データ数
        n_all = len(df_summary['target'].index)


        # しきい値のリストを作成し返却
        list_class_limit = []
        for i in range(num_of_class):
            # target_min から target_max までのデータ数をNで割ったi番目の累積データ数
            limit_idx = n_min_value + (n_all - n_min_value) * (float(i)/num_of_class)
            # 該当の累積データ数点での target の値を
            list_class_limit.append(df_summary['target'].iloc[int(limit_idx)])

        return list_class_limit


class TFMinibatch:

    #

    # trendデータを読み込んでメモリに格納
    def __init__(self, num_of_class):
        print("TFMinibatch init num_of_class:", num_of_class)

        self.CSV_DIR_PATH = tfs.TFSummary.TREND_FILE_DIR

        self.list_df_test = [] #正規化されたテスト用セット [1番目のclassのdfの配列, ... num_of_class番目のclassのdfの配列]
        self.list_df_train = [] #正規化された訓練用セット
        self.list_class_limit = [] #クラス分割のしきい値の配列
        self.num_of_class = num_of_class #クラス数

        # サマリデータとクラス数から、ラベルのしきい値を作成
        #self.list_class_limit = TFMinibatchHelper.calculateClassLimits(self.num_of_class)
        self.list_class_limit = TFMinibatchHelper.calculateClassLimitsFixed(self.num_of_class)
        print "list_class_limit"
        print self.list_class_limit

        # セットの初期化
        for i in range(num_of_class):
            self.list_df_test.append([])
            self.list_df_train.append([])

        # ファイルをすべて処理
        for filename in glob.glob(os.path.join(self.CSV_DIR_PATH, '*.csv')):

            # csvをロード
            df = pd.read_csv(filename, index_col="Date")

            # 正規化する
            #df = self.normalize(df)
            df = self.normalize_fixed(df)

            # ランダムに訓練とテストに分けてセットに格納
            self.append_to_target_class(df)

        # 逆順で入ってしまうのでreverseする
        self.list_df_test.reverse()
        self.list_df_train.reverse()
        #print len(self.list_df_train)
        #print len(self.list_df_test)
        #self.test_plot()

    def append_to_target_class(self, df):

        list_to_append = []
        # 訓練用セットとテスト用セットのどちらに追加するかを判定
        # ランダムで9:1で分類
        if np.random.rand() > 0.1:
            list_to_append = self.list_df_train
        else:
            list_to_append = self.list_df_test

        # クラスを判定するためにtargetを算出
        df_close = df['Adj Close']
        target = tfs.TFSummary.get_target(df_close)

        # 追加するリストの該当するクラスの配列へ追加

        #本番用
        #target_class = self.calculate_target_class(target)
        #テスト用
        target_class = self.calculate_target_class_fixed(df_close)

        list_to_append[target_class].append(df)

    def calculate_target_class(self, target):
        for i, val in enumerate(reversed(self.list_class_limit)):
            if target >= val:
                return i
        return 0

    # テスト用
    def calculate_target_class_fixed(self, df):
        hoge = df[60]
        if hoge > 1.5:
            return 4
        elif hoge > 0.5:
            return 3
        elif hoge > -0.5:
            return 2
        elif hoge > -1.5:
            return 1
        else:
            return 0


    #
    def get_next_batch_train(self, batch_size):
        return self.get_next_batch(batch_size, self.list_df_train)

    # 
    def get_next_batch_test(self, batch_size):
        return self.get_next_batch(batch_size, self.list_df_test)

    # batch_size で指定した数の訓練データセットを返却する
    # 訓練データセットの形式
    # x... 1x30
    # y... [0,0,0,0,1,0,...] one hot
    #      30日以降のmax/30日の値（正規化後）
    # batch_size はクラス数Nの整数倍になるように指定する
    # minibatch としては、ラベルが均等な数現れるように返却する
    def get_next_batch(self, batch_size, list_df):

        batch_x = []
        batch_y = []

        nloop = batch_size / self.num_of_class
        #print(batch_size)
        #print(self.num_of_class)
        #print(nloop)

        for iloop in range(nloop):
            for i, limit in enumerate(self.list_class_limit):

                # クラスiの配列からランダムにdfを取り出す
                df = random.choice(list_df[i])
                
                # inputの配列を作成
                # normalizedのみのため要素数は1
                list_input = []
                list_input.append((df['normalized'].values)[0:30])

                # 行列の入れ替えをして返却するリストに追加
                list_input_reshaped = zip(*list_input)
                batch_x.append(list_input_reshaped)

                # label
                #list_label = self.create_label(i)
                list_label = self.create_label_fixed(i)
                batch_y.append(list_label)

                #print len(df.index)
                #print tfs.TFSummary.get_target(df['Adj Close'])
                #print self.list_class_limit
                #print list_label
                #print list_input
                #print list_input_reshaped

        return batch_x, batch_y

    # 本番用
    def create_label(self, index):
        list_label = [0] * self.num_of_class
        list_label[index] = 1
        return list_label

    # テスト用
    def create_label_fixed(self, index):
        list_label = [0] * self.num_of_class
        if index == 0:
            list_label = [0.6,0.4,0,0,0]
        elif index == 1:
            list_label = [0.2,0.6,0.2,0,0]
        elif index == 2:
            list_label = [0,0.2,0.6,0.2,0]
        elif index == 3:
            list_label = [0,0,0.2,0.6,0.2]
        elif index == 4:
            list_label = [0,0,0,0.4,0.6]
        return list_label

    # データを正規化する
    # 開始から30日内での平均値で割って1を引いた列を追加する
    def normalize(self, df):

        # 0-30行の平均値
        mean = df['Adj Close'][0:29].mean()

        # 平均で割って1を引く
        df_adj_close_normalized = (df['Adj Close']/mean) - 1
        
        # "normalized"列を追加
        df = df.assign(normalized=df_adj_close_normalized)

        return df

    def normalize_fixed(self, df):
        # "normalized"列を追加
        df = df.assign(normalized=df['Adj Close'])
        return df

    # テストプロット
    def test_plot(self):
        for df_test in self.list_df_test:
            df_test['normalized'].plot()
            plt.show()
        
def test_tfminibatchHelper_calculateClassLimits():
    print(TFMinibatchHelper.calculateClassLimits(20))

def test_tfminibatch_init():
    print "not yet implimented"

def test_tfminibatch_get_next_batch():
    tfmb = TFMinibatch(10)
    tfmb.get_next_batch(10)

#test_minibatchHelper_calculateClassLimits()
#test_minibatch_init()
#test_tfminibatch_get_next_batch()

