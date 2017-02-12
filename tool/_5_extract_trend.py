# -*- coding: utf-8 -*-
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 設定
file_name = "./yahoo_data_lib/all.txt" # 入力ファイルパス
output_path = "./yahoo_data_trends/" # 出力フォルダパス

# パラメタ
rolling_mean_window = 50 # 移動平均を取る日数
trend_lower_limit = 30 # トレンドとみなす最低の期間



# 個別の銘柄データについて、トレンド範囲を切り出してcsv出力する
def extract_trend_for_brand(brand_id):

    # 返却するdataframe
    ret = None


    # 移動平均に使用する列名
    col_name = "Adj Close"

    # 個別銘柄データ
    df_brand = df[brand_id]

    # rolling_mean_window日の移動平均
    mean = df_brand[col_name].rolling(window=rolling_mean_window).mean()

    df_brand = df_brand.assign(Average = mean)

    # 終値と移動平均の差分
    diff = (df_brand[col_name] - mean)

    # df_brand['is_above_mean']に
    # 終値 >  移動平均 の場合 True
    # 終値 <= 移動平均 の場合 False を入れる
    max_diff = diff.max()
    min_diff = diff.min()
    norm = np.absolute(min_diff)+max_diff

    df_brand = df_brand.assign(is_above_mean = (diff / norm).apply(np.floor) == 0)

    # デバッグプロット
    # ((df_brand[col_name]/df_brand[col_name].mean()) - 1.5).plot(label="brand")
    # ((mean/df_brand[col_name].mean()) - 1.5).plot(label="mean")
    # (diff/norm).apply(np.floor).plot(label="abobe_mean")
    # plt.show()

    # df_brand['floor']でtrend_lower_limitの数以上1が続いている範囲を取り出す

    # トレンドの開始日付
    trend_start = None

    # 移動平均より大が継続している数のカウンタ
    trend_counter = 0

    # 銘柄ごとのトレンド通番
    brand_trend_index = 0

    for index, row in df_brand.iterrows():

        # 移動平均を上回るかどうか
        if row['is_above_mean']:

            # トレンド継続期間をインクリメント
            trend_counter = trend_counter + 1

            # トレンド開始日が未設定の場合、設定する
            if trend_start is None:
                trend_start = index

        else:

            # トレンド開始日〜この行までのデータ数が下限を超える場合
            # トレンド開始日〜この行までをトレンドとして取り出す
            # 注）最終日-1と最終日の間で移動平均と交わる
            if trend_counter >= trend_lower_limit:
                df_brand_trend = df_brand.loc[trend_start : index, ['Open','High','Low','Close','Volume', 'Adj Close', 'Average']]

                # デバッグプロット
                # df_brand_trend['Adj Close'].plot()
                # plt.show()
                # print df_brand_trend

                # 個別のcsvを出力
                csv_file_id = brand_id + "_" + str(trend_start)
                df_brand_trend.to_csv(output_path + csv_file_id + ".csv")


                # dataframeをjoinして返却しようとしたが、日付のindexが共通していないのでjoinできなかった

                # if ret is None:

                    # 返却するdataframeの初期作成
                    # ret = df_brand_trend
                    # ret.columns = [[csv_file_id] * len(ret.columns), ret.columns]

                # else:

                    # 返却するdataframeに追加
                    # df_brand_trend.columns = [[csv_file_id] * len(df_brand_trend.columns), df_brand_trend.columns]
                    # ret = ret.join(df_brand_trend)



            # トレンド開始日をリセット
            trend_start = None
            trend_counter = 0

    # return ret





# メイン処理
if __name__ == "__main__":
    
    # 出力データ
    # res = None

    # 全データ読み込み
    df = pd.read_csv(file_name, index_col=0, header=[0,1], parse_dates=[0])

    # 全銘柄について、extract_trend_for_brandを呼び出す
    for brand_id in df.columns.levels[0]:
        df_brand_trend_joined = extract_trend_for_brand(brand_id)

#        if df_brand_trend_joined is not None:
#
#
#            if res is None:
#                res = df_brand_trend_joined
#            else:
#                res = res.join(df_brand_trend_joined)
