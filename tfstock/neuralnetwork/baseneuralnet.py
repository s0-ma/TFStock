# -*- coding: utf-8 -*-
from abc import *

u""""""

class BaseNeuralNet(metaclass=ABCMeta):
    u"""ニューラルネットの基底クラス"""

    def __init__(self, minibatchcreator, run_parameter, network_parameter):
        u""""""
        self.minibatchcreator = minibatchcreator

    def train(self):
        u"""学習を実行します"""
        raise NotImplementedError

    def test(self):
        u"""検証を実行します"""
        raise NotImplementedError

    def save_result(self, dest):
        u"""学習の履歴データと学習結果パラメタを保存します"""
        raise NotImplementedError

    def show_result(self):
        raise NotImplementedError
