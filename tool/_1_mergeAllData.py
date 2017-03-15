#!/usr/bin/python

import glob
import pandas as pd

if __name__ == "__main__":

    file_list = glob.glob("../data/rawdata/*.txt")
    file_list.sort()

    f = file_list[0]
    ret = pd.read_csv(f, index_col="Date")
    ret.columns = [[f.split(".")[0]] * len(ret.columns), ret.columns]

    for f in file_list[1:]:
        df = pd.read_csv(f, index_col="Date")
        df.columns = [[f.split(".")[0]] * len(df.columns), df.columns]
        ret = ret.join(df)

    print(ret)

    ret.to_csv("../data/rawdata/merged/all.txt")

