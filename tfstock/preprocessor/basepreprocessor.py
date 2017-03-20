# -*- coding: utf-8 -*-
from abc import *
import pandas as pd

u"""dfの各列のデータを正規化、白色化し、ラベルを付けて返却するモジュール"""

class BasePreprocessor(metaclass=ABCMeta):
    u"""正規化、白色化クラスの基底クラス"""

    def __init__(self, df, parameter):
        self.df = df
        self.processed_df = None
        self.parameter = parameter

    def process(self):
        u"""正規化後、ラベル付けを実行します"""
        normalized_df = self.normalize(self.df, self.parameter)
        labeled_df = self.attachLabel(normalized_df, self.parameter)

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
        self.processed_df.to_csv(path)
