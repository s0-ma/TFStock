#!/usr/bin/python
# coding:utf-8

import time
import pandas
import datetime

def scraping_yahoo(code, start, end, term):
    print(code)
    base = "http://info.finance.yahoo.co.jp/history/?code={0}.T&{1}&{2}&tm={3}&p={4}"

    start = str(start)
    start = start.split("-")
    start = "sy={0}&sm={1}&sd={2}".format(start[0], start[1], start[2])
    end = str(end)
    end = end.split("-")
    end = "ey={0}&em={1}&ed={2}".format(end[0], end[1], end[2])
    page = 1

    result = []
    while True:
        url = base.format(code, start, end, term, page)
        df = pandas.read_html(url, header=0)
        if len(df[1]) == 0:
            break

        result.append(df[1])
        page += 1
    result = pandas.concat(result)
    result.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    #indexをyyyy/mm/ddに変換
    result = result.reset_index().drop("index", axis=1)
    for i in range(len(result)):
        result.loc[i, "Date"] = pandas.to_datetime(result["Date"][i].encode("utf-8").replace("年","/").replace("月","/").replace("日",""))
    result = result.set_index("Date")
    result = result.sort_index()

    #株式分割の表記文字列を削除
    result = result.dropna()

    return result


if __name__ == "__main__":
    nikkei_255=[1332,1333,1605,1721,1801,1802,1803,1808,1812,1925,
                1928,1963,2002,2269,2282,2432,2501,2502,2503,2531,
                2768,2801,2802,2871,2914,3086,3099,3101,3103,3105,
                3289,3382,3401,3402,3405,3407,3436,3861,3863,3865,
                4004,4005,4021,4042,4043,4061,4063,4151,4183,4188,
                4208,4272,4324,4452,4502,4503,4506,4507,4519,4523,
                4543,4568,4689,4704,4755,4901,4902,4911,5002,5020,
                5101,5108,5201,5202,5214,5232,5233,5301,5332,5333,
                5401,5406,5411,5413,5541,5631,5703,5706,5707,5711,
                5713,5714,5715,5801,5802,5803,5901,6103,6113,6301,
                6302,6305,6326,6361,6366,6367,6471,6472,6473,6479,
                6501,6502,6503,6504,6506,6508,6674,6701,6702,6703,
                6752,6758,6762,6767,6770,6773,6841,6857,6902,6952,
                6954,6971,6976,6988,7003,7004,7011,7012,7013,7186,
                7201,7202,7203,7205,7211,7261,7267,7269,7270,7272,
                7731,7733,7735,7751,7752,7762,7911,7912,7951,8001,
                8002,8015,8028,8031,8035,8053,8058,8233,8252,8253,
                8267,8303,8304,8306,8308,8309,8316,8331,8354,8355,
                8411,8601,8604,8628,8630,8725,8729,8750,8766,8795,
                8801,8802,8804,8830,9001,9005,9007,9008,9009,9020,
                9021,9022,9062,9064,9101,9104,9107,9202,9301,9412,
                9432,9433,9437,9501,9502,9503,9531,9532,9602,9613,
                9681,9735,9766,9983,9984]

    EndDate = datetime.date.today()
    StartDate = EndDate - datetime.timedelta(days=3000)

    for company in nikkei_255:
        data = scraping_yahoo(company, StartDate, EndDate, "d")
        try:
            data.to_csv("../data/rawdata/"+str(company)+".txt")
        except:
            print("err:" + str(company))

