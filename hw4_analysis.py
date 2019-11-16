import pandas as pd
import os
import numpy as np


# Init Dataframes
df = pd.read_csv("hw4-trainingset-pw2440.csv", dtype={"isic2l": object})
# Columns (50,172,255,256,257,258,268,280,376) have mixed types.

# count = raw_train.apply(lambda x: x.count(), axis=0)
# print(raw_train.append(count, ignore_index=True))

# count2 = df.apply(lambda col: col.isnull().sum() > 10000)


df = df.drop(df.columns[df.apply(lambda x: x.isnull().sum() > 10000)], axis=1)
# print(df.shape)
# print(df.dtypes)
# g = df.columns.to_series().groupby(df.dtypes).groups
# print(g)

print(df.filter(regex="wle_ca$", axis=1).columns)

def mask_wle_ca(df, column_name):
    mask20 = df[column_name] == 'Lowest to 20%'
    mask40 = df[column_name] == 'More than 20% to 40%'
    mask60 = df[column_name] == 'More than 40% to 60%'
    mask80 = df[column_name] == 'More than 60% to 80%'
    mask100 = df[column_name] == 'More than 80%
    
    df.loc[mask20, column_name] = 1
    df.loc[mask40, column_name] = 2
    df.loc[mask60, column_name] = 3
    df.loc[mask80, column_name] = 4
    df.loc[mask100, column_name] = 5

