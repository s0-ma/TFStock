# -*- coding: utf-8 -*-
import pandas as pd
from .basepreprocessor import *

u"""dfの各列のデータを前日比で正規化するモジュール"""

class RatioPreprocessor():

    def process(self, df):
        u"""
        各列のデータを前日比で置き換える
        一番目のデータは1とする
        """
        #for index, row in df.iterrows():

        print(df)
        for termid, subdf in df.groupby(level=0):
            for date, row in subdf.iterrows():
                # 一番目の場合、1を入れる
                # 二番目以降の場合、直前のデータで割った値を入れる
                print(row)
            break

