# -*- coding: utf-8 -*-
import pandas as pd
from .baserawdataprocessor import *

u"""CSVデータを処理し、pandasのdataframeを返却するモジュール"""

class FixedTermCSVProcessor(BaseRawDataProcessor):
    u"""固定長期間のCSVデータ処理クラス"""

    def add_col_to(self, src, df):
        raise NotImplementedError

    def create_df(self, src):
        u"""
        srcで渡されたパスのCSVファイルをよみこみ、dfとして返却する
        パス直下のCSVデータの列数、行数はすべて等しい前提
        Args:
            src: CSVファイルパス
        """

        # csvをロードし、dfに入れる
        csv_file_path = src
        df = pd.read_csv(csv_file_path, index_col=["TermID", "Date"])

        return df

class EarningCSVProcessor(BaseRawDataProcessor):
    u"""決算データのCSVデータ処理クラス"""

    def add_col_to(self, src, df):
        u"""
        srcで渡されたパスのCSVデータをよみこみ、読み込んだcsvのデータを渡されたdfに追加する。
        決算データは決算日のみデータが存在するため、間の期間は同じ値で埋める。
        Args:
            src: CSVファイルのあるディレクトリパス
            df: 処理データを追加する対象のdf
        """
        return df

    def create_df(self, src):
        raise NotImplementedError





