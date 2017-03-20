# -*- coding: utf-8 -*-
import pandas as pd
from .basepreprocessor import *

u"""dfの各列のデータを前日比で正規化するモジュール"""

class RatioPreprocessor(BasePreprocessor):

    def __init__(self, df, parameter):
        super.__init__(df, parameter)
        self.list_threshold = []

    def normalize(self):
        u"""
        各列のデータを前日比で置き換える
        一番目のデータは1とする
        parameterは使用しない。
        """
        #for index, row in df.iterrows():
        df = self.df

        print(df)
        for termid, subdf in df.groupby(level=0):
            for date, row in subdf.iterrows():
                # 一番目の場合、1を入れる
                # 二番目以降の場合、直前のデータで割った値を入れる
                print(row)
            break

    def attach_label(self):
        u"""
        parameter.target_column
        parameter.startdate_index
        parameter.enddate_index
        parameter.ratio_min
        parameter.ratio_max
        parameter.num_of_class
        target_column列のstartDateIdx日目のデータと、endDateIdx日目のデータの比を使用する
        比の想定最小値ratio_min、最大値ratio_maxをnum_of_classで等分し、ラベルにする
        比の想定最小値未満、最大値を超えるものは端のクラスに追加する
        """
        df = self.df
        parameter = self.parameter
        for termid, subdf in df.groupby(level=0):
            start = subdf[parameter.target_column][parameter.startdate_index]
            end = subdf[parameter.target_column][parameter.enddate_index]


    def get_threshold(self):
        u"""attach_labelで割りつけるラベルの、対象データに対するスレッショルド値を取得する"""
        if self.list_threshold:
            return self.list_threshold
        else:
            raise NotImplementedError
            return ""


class RatioPreprocessorParameter:
    u"""RatioPreprocessorで使用するパラメタを定義します"""

    def __init__(self, startdate_index, enddate_index):
        self.startdate_index = startdate_index
        self.enddate_index = enddate_index

    def __init__(self, parameter):
        self.startdate_index = parameter.startdate_index
        self.enddate_index = parameter.enddate_index

