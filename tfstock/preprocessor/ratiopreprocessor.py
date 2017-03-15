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
        for index, row in df.iterrows():
            print(index, row)

