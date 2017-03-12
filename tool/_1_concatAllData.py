#!/usr/bin/python

import glob
import pandas as pd
import os.path

if __name__ == "__main__":

    file_list = glob.glob("../data/rawdata/*.txt")
    file_list.sort()

    list_df = []
    list_key = []

    print(file_list)

    for f in file_list:
        print(f)
        df = pd.read_csv(f, index_col="Date")
        list_df.append(df)
        list_key.append(os.path.basename(f).split(".")[0])

    print(list_key)

    ret = pd.concat(list_df, keys=list_key)
    ret.index.names = ['Brand', 'Date']

    ret.to_csv("../data/rawdata/marged/concat.txt")

