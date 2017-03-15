# -*- coding: utf-8 -*-
from abc import *

u"""生データを処理し、pandasのdataframeを返却するモジュール"""

class BaseRawDataProcessor(metaclass=ABCMeta):
    u"""生データ処理クラスの基底クラス"""

    @abstractmethod
    def add_col_to(self, src, df):
        u"""srcから生データをよみこみ、引数として渡されたdfに
        読み込んだcsvのデータを新しい列として追加する"""
        raise NotImplementedError

    @abstractmethod
    def create_df(self, src):
        u"""srcから生データをよみこみ、読み込んだcsvのデータをdfとして返却する"""
        raise NotImplementedError

