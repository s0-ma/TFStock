# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from tfstock.rawdataprocessor.csvprocessor import *
from tfstock.preprocessor.ratiopreprocessor import *

csvprocessor =  FixedTermCSVProcessor()
preprocessor = RatioPreprocessor()

df = csvprocessor.create_df('../data/quadratic_0.1/')

preprocessed_df = preprocessor.process(df)
