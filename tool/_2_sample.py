#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd

#"all.txt"を 0列目をindexとして, 0,1行目をheaderとして 読み込む。0番目(Date)はdatetimeでパース。
df = pd.read_csv("../data/rawdata/all.txt", index_col=0, header=[0,1], parse_dates=[0])
print type(df.index[0])

#列ラベル1が"1332" かつ 列ラベル2が"Adj Close"のものをprint
print df["1332"]["Adj Close"]

#axis=1(列)のラベルに対し、 その2番目の階層のラベルの中で、"Adj Close"ラベルを持つものをprint
print df.xs("Adj Close", level=1, axis=1)
