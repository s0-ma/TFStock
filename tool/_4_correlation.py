#! /usr/bin/python
# -*- coding:utf-8 -*-

import datetime
import pandas as pd
import matplotlib.pyplot as plt
#pd.set_option("display.width", 200)

D_BACK = 100
D_FOWARD = 0

def partialAutoCorrelation(df1, df2):

    df1.columns = [s + "_x" for s in df1.columns]
    df2.columns = [s + "_y" for s in df2.columns]

    x = [ ]
    y = [ ]
    ret = pd.DataFrame({"day":range(-D_BACK,D_FOWARD)},columns=["day","corr"]).set_index("day")

    for i in range(-D_BACK,D_FOWARD):
        df2_shift = df2.copy()
        df2_shift.index = df2.index +  datetime.timedelta(days=i)
        out = pd.concat([df1, df2_shift], axis=1)
        y.append(out.corr().ix[0,1])

    ret["corr"] = y

    return ret

def changeValuesToRatio(series):
    series_orig = series.reset_index()

    series_dif = series_orig.copy()
    series_dif.index = series_dif.index + 1

    series_orig["Adj Close"] = series_orig["Adj Close"]/series_dif["Adj Close"]
    series_orig = series_orig.set_index("Date")
    return series_orig


if __name__ == "__main__":

    id1 = "8316"
    id2 = "8411"

    df = pd.read_csv("all.txt", index_col=0, header=[0,1], parse_dates=[0])

    df1 = df[id1][["Adj Close",]]
    df2 = df[id2][["Adj Close",]]

    #前日比に関する相関が見たいなら、コメントアウト
    #df1 = changeValuesToRatio(df1)
    #df2 = changeValuesToRatio(df2)

    ret = partialAutoCorrelation(df1, df2)
    ret.plot()

    df = pd.concat([df1.iloc[:,0], df2.iloc[:,0]], axis=1)
    df.columns = [id1,id2]
    pd.scatter_matrix(df)

    plt.show()
