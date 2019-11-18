import pandas as pd
import os
import numpy as np
import unicodedata

# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Integer
# 'age_r', 'v231', 'v272', 'earnmthbonusppp', 'readytolearn', 'icthome',
#       'ictwork', 'influence', 'planning', 'readhome', 'readwork', 'taskdisc',
#       'writhome', 'writwork'
#
# replace (ordinal or binary categories)
# v200, fnfe12jr, edcat7
#
# one hot
# ["v31", "ctryqual"]
#


# Init Dataframes
df = pd.read_csv("hw4-trainingset-pw2440.csv")
# Columns (50,172,255,256,257,258,268,280,376) have mixed types.
print(df.select_dtypes(include=["float64", "int64"]))

selected_features = [
    "age_r",
    "v200",
    "v231",
    "v272",
    "v31",
    # "ctryqual",
    "yrsqual",
    "earnmthbonusppp",
    "earnhrbonusppp",
   "nfe12jr",
   "nfe12njr",
   "fnfe12jr",
   "fnfaet12jr",
   "fnfaet12njr",
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
    # "v200": {
    #     "High school": 2.0,
    #     "Above high school": 3.0,
    #     "Less than high school": 1.0,
    #     "Not definable": np.nan,
    # },
    "fnfe12jr": {
        "Participated in FE or NFE for JR reasons": 1.0,
        "Did not participate in FE or NFE for JR reasons": 0.0,
    },
    "nfe12jr": {
        "Participated in NFE for JR reasons": 1,
        "Did not participate in NFE for JR reasons": 0,
    },
    "nfe12njr": {
        "Did not participate in NFE for NJR reasons": 0,
        "Participated in NFE for NJR reasons": 1,
    },
    "fnfaet12jr": {
        "Participated in formal or non-formal AET for JR reasons": 1,
        "Did not participate in formal or non-formal AET for JR reasons" : 0,
    },
    "fnfaet12njr":{
        "Did not participate in formal or non-formal AET for non JR reasons": 0,
        "Participated in formal or non-formal AET for non JR reasons" : 1,
    },
    "edcat7": {
        "Primary or less (ISCED 1 or less)": 1.0,
        "Lower secondary (ISCED 2, ISCED 3C short)": 2.0,
        "Upper secondary (ISCED 3A-B, C long)" : 3.0,
        "Post-secondary, non-tertiary (ISCED 4A-B-C)" : 4.0,
        "Tertiary  bachelor degree (ISCED 5A)" : 5.0,
        "Tertiary - bachelor/master/research degree (ISCED 5A/6)" : 6.0,
        "Tertiary  master/research degree (ISCED 5A/6)" : 7.0,
        "Tertiary  professional degree (ISCED 5B)" : 8.0,
    }
}

df1 = df[selected_features].copy()
df1['edcat7'] = df1['edcat7'].str.replace('[^\x00-\x7F]','')
int_df = df1.select_dtypes(include=["float64", "int64"]).copy()

# fill mean in integer columns
int_df.fillna(int_df.mean(), inplace=True)

ord_cat = ["nfe12jr", "nfe12njr", "fnfaet12jr", "fnfaet12njr", "fnfe12jr"]
#cat_obj = ["v31", "ctryqual", "edcat7", "v200"]
cat_obj = ["v31", "v200", "edcat7"] + ord_cat
obj_df = df1.select_dtypes(include=["object"]).copy()

obj_df.replace(cleanup, inplace=True)

# fill in median
for i in ord_cat:
    obj_df[i].fillna(obj_df[i].median(), inplace=True)

# # fill in mode
for i in cat_obj:
    obj_df[i].fillna(obj_df[i].mode()[0], inplace=True)

# obj_df = obj_df.astype({'edcat7': 'int32', 'v200': 'int32', 'fnfe12jr': 'int32'})
# obj_df = obj_df.astype({'v200': 'int32', 'fnfe12jr': 'int32'})

# one hot
obj_df = pd.get_dummies(obj_df, columns=cat_obj)
df_init = pd.concat([int_df, obj_df], axis=1)

# df_init = int_df

# print(df_init)
target = df["job_performance"]
print(target.median())

# train and test set
# x_train = df_init[:-4000]
# x_test = df_init[-4000:]

# y_train = target[:-4000]
# y_test = target[-4000:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(df_init_x_train, df_init_y_train)

# # Make predictions using the testing set
# df_y_pred = regr.predict(df_init_x_test)

# # The coefficients
# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(df_init_y_test, df_y_pred))
# # Explained variance score: 1 is perfect prediction
# print("Variance score: %.2f" % r2_score(df_init_y_test, df_y_pred))

##### gradient boosted alg #######
# from sklearn import ensemble
# from sklearn.metrics import mean_squared_error

# # Fit regression model
# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#           'learning_rate': 0.01, 'loss': 'ls'}
# clf = ensemble.GradientBoostingRegressor(**params)

# clf.fit(x_train, y_train)
# mse = mean_squared_error(y_test, clf.predict(x_test))
# print("MSE: %.4f" % mse)
####################################

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
#     kfold = KFold(n_splits=10, random_state=15)
#     cv_results = cross_val_score(
#         model, x_train, y_train, cv=kfold, scoring="neg_mean_squared_error"
#     )
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
