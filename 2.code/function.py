from const import *
from const import *
from function import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
import itertools

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from tqdm import tqdm
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from catboost import Pool

warnings.simplefilter("ignore")


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def creansing_dataset(df):
    # year
    is_error_year = df["year"] > 2023
    df.loc[is_error_year, "year"] = df[is_error_year]["year"] - 1000
    # size
    df["size"] = df["size"].replace(size_replace)
    # manufacturer
    df["manufacturer"] = df["manufacturer"].replace(manufactuer_replace)
    # odometer
    is_odo_error = df["odometer"] < -1
    is_odo_unknown = df["odometer"] == -1
    df.loc[is_odo_error, "odometer"] = df[is_odo_error]["odometer"] * (-1)
    df.loc[is_odo_unknown, "odometer"] = np.nan

    return df


def add_odometer_price_tile(df, cat_cols):
    f_odometer = df["odometer"] < 400000
    df.loc[f_odometer, "bins"] = pd.cut(df[f_odometer]["odometer"], bins=30)
    for col in cat_cols + ["year"]:
        df_tile = (
            df.groupby(["bins", col])
            .agg(
                odo_25_pertile=("price", lambda sr: sr.quantile(0.25)),
                odo_50_pertile=("price", lambda sr: sr.quantile(0.5)),
                odo_75_pertile=("price", lambda sr: sr.quantile(0.75)),
                odo_std=("price", "std"),
            )
            .add_prefix(f"{col}_")
            .reset_index()
        )

        df = df.merge(df_tile, on=["bins", col], how="left")

    return df.drop("bins", axis=1)


def create_num_features(df, cat_cols):
    df["elapsed_year"] = 2023 - df["year"]
    df = add_odometer_price_tile(df, cat_cols)

    del df["year"]

    return df


def create_cat_features(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].factorize()[0]
        df[col] = df[col].astype("category")
    return df


def create_features_main(df, cat_cols):
    df = create_num_features(df, cat_cols)
    df = create_cat_features(df, cat_cols)

    return df


def built_catboost_model(model_type, cat_features, Xt, yt, Xe, ye):
    if model_type == "Classification":
        model = CatBoostClassifier(
            learning_rate=0.1,
            iterations=500,
            loss_function="Logloss",
            early_stopping_rounds=10,
            random_seed=42,
        )
    if model_type == "Regression":
        model = CatBoostRegressor(
            learning_rate=0.1,
            iterations=500,
            loss_function="MAPE",
            early_stopping_rounds=10,
            random_seed=42,
        )

    model.fit(
        X=Xt,
        y=yt,
        eval_set=(Xe, ye),
        cat_features=cat_features,
        use_best_model=True,
        verbose=100,
    )

    return model


def mape(true, pred):
    return np.mean(np.abs((pred - true) / true))


def plot_result(preds, cv):
    pred_df_l = []
    for i in range(cv):
        pred_df_l.append(
            pd.DataFrame(np.array(preds[i]).T, columns=["id", "actual", "pred"])
        )

    pred_df = pd.concat(pred_df_l)

    plt.figure(figsize=(8, 5), facecolor="azure", edgecolor="coral", linewidth=2)
    bins = np.linspace(0, pred_df["actual"].max(), 200)
    plt.hist(pred_df["actual"], bins, alpha=0.5, label="actual")
    plt.hist(pred_df["pred"], bins, alpha=0.5, label="b")

    fig = plt.figure(figsize=(5, 5), facecolor="azure", edgecolor="coral", linewidth=2)
    ax1 = fig.add_subplot()
    ax1.set_ylim([0, pred_df["actual"].max()])
    ax1.set_xlim([0, pred_df["actual"].max()])
    ax1.scatter(pred_df["actual"], pred_df["pred"])

    # fig = plt.figure(figsize=(5, 5), facecolor="azure", edgecolor="coral", linewidth=2)
    # ax1 = fig.add_subplot()
    # ax1.set_ylim([0, np.exp(pred_df["actual"]).max()])
    # ax1.set_xlim([0, np.exp(pred_df["pred"]).max()])
    # ax1.scatter(np.exp(pred_df["actual"]), np.exp(pred_df["pred"]))


def plot_importance(importnace_l, input_cols, show_n=50):
    imp_df = pd.DataFrame(index=input_cols)
    for i in range(len(importnace_l)):
        imp_df[f"cv{i+1}"] = importnace_l[i]

    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean").tail(show_n)

    plt.figure(figsize=(8, 12), facecolor="azure", edgecolor="coral", linewidth=2)
    plt.boxplot(
        [imp_df.T[col].values for col in imp_df.index], labels=imp_df.index, vert=False
    )


def calc_confusion_matricx(preds, cv):
    pred_df_l = []
    for i in range(cv):
        pred_df_l.append(
            pd.DataFrame(np.array(preds[i]).T, columns=["id", "actual", "pred"])
        )

    pred_df = pd.concat(pred_df_l)
    print(f"f1 score = {f1_score(pred_df['actual'], pred_df['pred'])}")

    return pd.DataFrame(
        confusion_matrix(pred_df["actual"], pred_df["pred"]),
        columns=["Negative_Predict", "Positive_Predict"],
        index=["Negative_Actual", "Positive_Actual"],
    )


def train_model(train_df, p_col, cv, m_type, cat_cols, input_cols):
    kf = KFold(n_splits=cv)

    models = []
    preds = []
    scores = []
    importnace_l = []

    for train_index, valid_index in kf.split(train_df):
        x_train = train_df.iloc[train_index][input_cols]
        y_train = train_df.iloc[train_index][p_col]
        x_valid = train_df.iloc[valid_index][input_cols]
        y_valid = train_df.iloc[valid_index][p_col]

        model = built_catboost_model(
            m_type, cat_cols, x_train, y_train, x_valid, y_valid
        )

        valid_pred = model.predict(x_valid)

        scores.append(mape(y_valid, valid_pred).round(5))
        preds.append([valid_index, y_valid.values, valid_pred])
        models.append(model)
        importnace_l.append(model.get_feature_importance())

        del x_train, y_train, x_valid, y_valid, model

    return models, preds, scores, importnace_l


def add_c_price(test_df, cv, input_cols, models):
    for i in range(cv):
        test_df[f"fold{i}"] = models[i].predict(test_df[input_cols])

    test_df["sum"] = test_df[[f"fold{i}" for i in range(cv)]].sum(axis=1)
    test_df.loc[test_df["sum"] < 3, "c_price"] = 0
    test_df.loc[test_df["sum"] >= 3, "c_price"] = 1

    return test_df
