import pandas as pd
import glob
import re
import itertools as it
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, max_error


def concate(path, model):
    ini = str(path) + "/" + str(model) + "_1" + ".csv"
    df_ = pd.read_csv(ini)
    for filename in glob.glob(str(path) + "/" + str(model) + "_" + "*.csv"):
        if filename != ini:
            k = filename.split(str(model))[1].split(".")[0].split("_")[1]
            df = pd.read_csv(filename)
            df_ = df_.join(df.set_index(["epoch"]), on="epoch", rsuffix=str(k))
    return df_


def operation_maker(des, total):
    des_prom = [de + "_prom" for de in des]
    num_ = [str(i) for i in range(2, total + 1)] + [""]
    des_num = [[de + num for num in num_] for de in des]
    phrase = ["+".join(des) for des in des_num]
    return phrase, des_prom


def iden_op(phrase, name="df"):
    ope = re.split(r'[\+\-*/]', phrase)
    simb = re.findall(r'[\+\-*/]', phrase)
    new_c = ""
    while len(simb) != len(ope):
        simb.append("")
    for part, simbol in zip(ope, simb):
        try:
            float(part)
            new_c += f"{part}{simbol}"
        except ValueError:
            new_c += f"{name}['{part}']{simbol}"
    return new_c


def smooth(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def prom(path, model, des=["loss", "mae", "val_mae", "val_loss"], total=10):
    df = concate(path, model)
    suma, prom_s = operation_maker(des, total)
    for valor, sumat in zip(prom_s, suma):
        df[valor] = (eval(iden_op(sumat))) / total
    return df


def comb_des(df, n_des):
    comb = np.array(list(it.combinations(df.columns, n_des)))
    return comb


def corr(df, y_name, n_des, r_2_ref, r_cv_ref=-10, cv_n=2):
    y = df[y_name].to_numpy()
    df = df.drop(columns=y_name)
    df = df.select_dtypes(include=np.number)
    comb = comb_des(df, n_des)
    for column in comb:
        X = df[column].to_numpy()
        model = LinearRegression().fit(X, y)
        y_pre, r_2, scores = model.predict(X), model.score(
            X, y), cross_val_score(model, X, y, cv=cv_n)
        mae, max_ = mean_absolute_error(y, y_pre), max_error(y, y_pre)
        f, p = f_regression(X, y)
        if r_2 > r_2_ref:
            print(column, "R2:", r_2, "CV:",
                  np.mean(scores), "F:", f, "MAE:", mae, "Max:", max_)
