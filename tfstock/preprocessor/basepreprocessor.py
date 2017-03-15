# -*- coding: utf-8 -*-
from abc import *

u"""dfの各列のデータを正規化、白色化するモジュール"""

class BasePreprocessor(metaclass=ABCMeta):
    u"""正規化、白色化クラスの基底クラス"""

    def process(self, df):
        u"""引数で渡されたdfを正規化、白色化して返却する"""
        raise NotImplementedError

