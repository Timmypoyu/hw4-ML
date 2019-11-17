import pandas as pd
import os
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# from sklearn import datasets
# from sklearn.feature_selection import RFE
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import GradientBoostingRegressor


# Init Dataframes
df = pd.read_csv("hw4-trainingset-pw2440.csv")
# Columns (50,172,255,256,257,258,268,280,376) have mixed types.

selected_features = [
    "age_r",
    "v200",
    "v231",
    "v272",
    "v31",
    "ctryqual",
    "earnmthbonusppp",
    "fnfe12jr",
    "readytolearn",
    "icthome",
    "ictwork",
    "influence",
    "planning",
    "readhome",
    "readwork",
    "taskdisc",
    "writhome",
    "writwork",
    "edcat7",
]

cleanup = {
    "v200": {
        "High school": 2.0,
        "Above high school": 3.0,
        "Less than high school": 1.0,
    },
    "fnfe12jr": {
        "Participated in FE or NFE for JR reasons": 1.0,
        "Did not participate in FE or NFE for JR reasons": 0.0,
    },
}

df1 = df[selected_features]
int_df = df1.select_dtypes(include=["float64", "int64"]).copy()
# Integer
# 'age_r', 'v231', 'v272', 'earnmthbonusppp', 'readytolearn', 'icthome',
#       'ictwork', 'influence', 'planning', 'readhome', 'readwork', 'taskdisc',
#       'writhome', 'writwork'
#
# replace (ordinal or binary categories)
# v200, fnfe12jr
#
# one hot
# ["v31", "ctryqual"]
#

# fill mean in integer columns
int_df.fillna(int_df.mean(), inplace=True)

# object columns setup
cat_obj = ["v31", "ctryqual", "v200", "fnfe12jr", "edcat7"]
obj_df = df1.select_dtypes(include=["object"]).copy()
# fill in mode
for i in cat_obj:
    obj_df[i].fillna(obj_df[i].mode()[0], inplace=True)

obj_df.replace(cleanup, inplace=True)
# one hot
obj_df = pd.get_dummies(obj_df, columns=cat_obj)
df_init = pd.concat([int_df, obj_df], axis=1)
target = df["job_performance"]

# print(df_init.isnull().sum())

df_init_x_train = df_init[:-4000]
df_init_x_test = df_init[-4000:]

df_init_y_train = target[:-4000]
df_init_y_test = target[-4000:]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(df_init_x_train, df_init_y_train)

# Make predictions using the testing set
df_y_pred = regr.predict(df_init_x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(df_init_y_test, df_y_pred))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(df_init_y_test, df_y_pred))

######
# pipelines = []
# pipelines.append(
#     ("ScaledLR", Pipeline([("Scaler", StandardScaler()), ("LR", LinearRegression())]))
# )
# pipelines.append(
#     ("ScaledLASSO", Pipeline([("Scaler", StandardScaler()), ("LASSO", Lasso())]))
# )
# pipelines.append(
#     ("ScaledEN", Pipeline([("Scaler", StandardScaler()), ("EN", ElasticNet())]))
# )
# pipelines.append(
#     (
#         "ScaledKNN",
#         Pipeline([("Scaler", StandardScaler()), ("KNN", KNeighborsRegressor())]),
#     )
# )
# pipelines.append(
#     (
#         "ScaledCART",
#         Pipeline([("Scaler", StandardScaler()), ("CART", DecisionTreeRegressor())]),
#     )
# )
# pipelines.append(
#     (
#         "ScaledGBM",
#         Pipeline([("Scaler", StandardScaler()), ("GBM", GradientBoostingRegressor())]),
#     )
# )

# results = []
# names = []
# for name, model in pipelines:
#     kfold = KFold(n_splits=10, random_state=21)
#     cv_results = cross_val_score(
#         model, df_init, target, cv=kfold, scoring="neg_mean_squared_error"
#     )
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)

# count = raw_train.apply(lambda x: x.count(), axis=0)
# print(raw_train.append(count, ignore_index=True))

# count2 = df.apply(lambda col: col.isnull().sum() > 10000)


# df = df.drop(df.columns[df.apply(lambda x: x.isnull().sum() > 18000)], axis=1)
# print(df.shape)
# print(df.dtypes)
# g = df.columns.to_series().groupby(df.dtypes).groups
# print(g)

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

