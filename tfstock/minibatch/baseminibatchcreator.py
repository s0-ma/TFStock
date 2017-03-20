# -*- coding: utf-8 -*-
from abc import *

u"""正規化済みのdfを読み込み、ミニバッチを返却するモジュール"""

class BaseMinibatchCreator(metaclass=ABCMeta):
    u"""ミニバッチ返却クラスの基底クラス"""

    def __init__(self, df, parameter):
        self.df = df

    def get_next_batch(self, batch_size):
        u"""ミニバッチを返却する"""
        raise NotImplementedError

