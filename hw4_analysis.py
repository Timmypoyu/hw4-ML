import pandas as pd
import os
import numpy as np


# Init Dataframes
df = pd.read_csv("hw4-trainingset-pw2440.csv")
# Columns (50,172,255,256,257,258,268,280,376) have mixed types.

# count = raw_train.apply(lambda x: x.count(), axis=0)
# print(raw_train.append(count, ignore_index=True))

# count2 = df.apply(lambda col: col.isnull().sum() > 10000)


# df = df.drop(df.columns[df.apply(lambda x: x.isnull().sum() > 18000)], axis=1)
# print(df.shape)
# print(df.dtypes)
# g = df.columns.to_series().groupby(df.dtypes).groups
# print(g)

# cut out country 
#df.drop(columns=['cntryid', 'cntryid_e'])
cleanup = { 'v200' : {'High school': 0, 'Above high school': 1},
'fnfe12jr': {'Participated in FE or NFE for JR reasons': 1, 'Did not participate in FE or NFE for JR reasons': 0 }}

selected_features = ['age_r', 'v200', 'v231', 'v272', 'v31', 'ctryqual'
,'earnmthbonusppp' , 'fnfe12jr', 'readytolearn' , 'icthome', 'ictwork', 'influence', 'planning',
'readhome', 'readwork', 'taskdisc', 'writhome', 'writwork', 'edcat7']

df1 = df[selected_features]
int_df = df1.select_dtypes(include=['float64', 'int64'])
print(int_df.columns)
int_df.fillna(int_df.mean(), inplace = True)
print(int_df)

obj_df = df1.select_dtypes(include=['object'])
obj_df.replace(cleanup, inplace=True)
cat_obj = ['v31', 'ctryqual']

for i in cat_obj:
   obj_df[i].fillna(obj_df[i].mode()[0], inplace=True)

pd.get_dummies(obj_df, columns=cat_obj)

print(obj_df)

#print(df1)

# obj_df = df.select_dtypes(include=['float64', 'int64']).copy()
# print(obj_df.columns)

# def mask_wle_ca(df, column_name):
#     mask20 = df[column_name] == 'Lowest to 20%'
#     mask40 = df[column_name] == 'More than 20% to 40%'
#     mask60 = df[column_name] == 'More than 40% to 60%'
#     mask80 = df[column_name] == 'More than 60% to 80%'
#     mask100 = df[column_name] == 'More than 80%'
    
#     df.loc[mask20, column_name] = 1
#     df.loc[mask40, column_name] = 2
#     df.loc[mask60, column_name] = 3
#     df.loc[mask80, column_name] = 4
#     df.loc[mask100, column_name] = 5

# def binary_gender(df, column_name):
#     mask_male = df[column_name] == 'Male'
#     mask_female = df[column_name] == 'Female'

#     df.loc[mask_male, column_name] = 0
#     df.loc[maks_female, column_name] = 1

# cleanup_wle_ca = { wle_ca_df: 

# wle_ca_df = df.filter(regex="wle_ca$", axis=1).copy()
# wle_ca_list = list(df.filter(regex="wle_ca$", axis=1).columns)

# df_2 = df.copy()
# for i in wle_ca_list:
# 	mask_wle_ca(df_2, i)

# print(df_2)




