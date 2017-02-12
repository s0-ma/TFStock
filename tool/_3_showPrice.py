#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../data/rawdata/all.txt", index_col=0, header=[0,1], parse_dates=[0])

#sample: 全部plot
#df.xs("Adj Close", level=1, axis=1).plot()

ax = df["6701"]["Adj Close"].plot(label="6701")
#df["1333"]["Adj Close"].plot(ax=ax, label="1333")

plt.show()

