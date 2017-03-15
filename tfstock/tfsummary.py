# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class TFSummary:

    SUMMARY_FILE_PATH = '../data/other/trend_summary.csv'
    TREND_FILE_DIR = '../data/quadratic_0.5'

    # 機械学習で当てに行く値の返却
    @classmethod
    def get_target(cls, df):
        val_of_30day = df[30]
        max_after_30day = df.iloc[30:].max()
        target = max_after_30day / val_of_30day
        return target

    # trend の csv ファイルをすべて読んでサマリのdfをcsv出力
    @classmethod
    def export_summary_csv(cls):
        # サマリ作成
        summary_df = pd.DataFrame(columns=('max', 'min', 'val_of_30day', 'max_after_30day', 'target', 'length'))
    
        # ファイルをすべて処理
        path = TFSummary.TREND_FILE_DIR

        i = 0
        for filename in glob.glob(os.path.join(path, '*.csv')):
            df = pd.read_csv(filename, index_col="Date")
            df_close = df['Adj Close']
            target = TFSummary.get_target(df_close)
            maxval = df_close.max()
            minval = df_close.min()
            val_of_30day = df_close[30]
            max_after_30day = df_close.iloc[30:].max()
            summary_df.loc[i] = [maxval, minval, val_of_30day, max_after_30day, target ,len(df.index)]
            #df["Adj Close"].plot(label="Adj Close")
            #df["Average"].plot(label="Average")
        
            #plt.show()
        
            i = i + 1
        
        summary_df.to_csv(TFSummary.SUMMARY_FILE_PATH)
    
if __name__ == "__main__":
    tfs = TFSummary()
    tfs.export_summary_csv()
