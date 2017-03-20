# -*- coding: utf-8 -*-
from abc import *
import pandas as pd

u"""dfの各列のデータを正規化、白色化し、ラベルを付けて返却するモジュール"""

class BasePreprocessor(metaclass=ABCMeta):
    u"""正規化、白色化クラスの基底クラス"""

    def __init__(self, df, parameter):
        self.df = df
        self.labeled_df = None
        self.parameter = parameter

    def process(self):
        df = self.df
        parameter = self.parameter
        normalized_df = self.normalize(df, parameter)
        labeled_df = self.attachLabel(normalized_df, parameter)
        self.processed_df = labeled_df
        return labeled_df;

    def normalize(self):
        u"""引数で渡されたdfを正規化、白色化して返却する"""
        raise NotImplementedError

    def attach_label(self):
        u"""引数で渡されたdfにラベル列を追加して返却する"""
        raise NotImplementedError

    def save_csv(self, path):
        u"""正規化し、ラベル付けしたdfをcsvに保存する"""
        self.labeled_df.to_csv(path)
