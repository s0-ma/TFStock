# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from tfstock.common.applicationparameter import *
from tfstock.common.constant import *
from tfstock.rawdataprocessor.csvprocessor import *
from tfstock.preprocessor.ratiopreprocessor import *
from tfstock.minibatch.simpleminibatchcreator import *
from tfstock.neuralnetwork.lstm import *

# 1.アプリケーションのパラメタ設定
parameter = ApParameter(startdate_index=30,
                        enddate_index=60)

# 2.収集した生データをCSVから読込
csvprocessor =  FixedTermCSVProcessor()
df = csvprocessor.create_df('../data/quadratic_0.1/merged/concat.txt')

# 3.事前データ処理

# 3-1.前日比データを使って正規化し、ラベル付け
preprocessor_parameter = RatioPreprocessorParameter(parameter)
preprocessor = RatioPreprocessor(df, parameter=preprocessor_parameter)
df_preprocessed = preprocessor.process()

# 3-2.学習用データをCSVに保存
preprocessor.save_csv('../work/quadratic_0.1_processed.txt')

# 4.学習

# 4-1.学習用データからミニバッチを生成
minibatch_parameter = SimpleMinibatchCreatorParameter(parameter)
minibatch_creator = SimpleMinibatchCreator(minibatch_parameter)

# 4-2.学習用パラメタ設定

# 4-3.学習・テスト
neural_network = None
neural_network.train()
neural_network.test()

# 4-4.学習結果を保存
neural_network.save("../work/quadratic_0.1_result.txt")

# 4-5.可視化
neural_network.show_result()
