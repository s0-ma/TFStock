#!/usr/bin/python

import glob
import pandas as pd
import os.path
import sys

if __name__ == "__main__":

    args = sys.argv
    print(args)
    if len(args) < 3:
        raise ValueError("usage: python _1_concatAllData.py ../data/rawdata/  *.txt")

    print(args[1]+args[2])
    file_list = glob.glob(args[1]+ args[2])
    file_list.sort()

    list_df = []
    list_key = []

    print(file_list)

    for f in file_list:
        df = pd.read_csv(f, index_col="Date")
        list_df.append(df)
        list_key.append(os.path.basename(f).split(".")[0])

    print(list_key)

    ret = pd.concat(list_df, keys=list_key)
    ret.index.names = ['TermID', 'Date']

    ret.to_csv(args[1] + "merged/concat.txt")

