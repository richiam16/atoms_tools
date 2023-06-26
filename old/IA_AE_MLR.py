from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ie_ae = pd.read_csv("/mnt/c/Users/richi/Desktop/QTAIM/NPA/IE_AE_DNN_2.csv")
# ie_ae = ie_ae[(ie_ae["IE_AE"] < 20)]
y = ie_ae[["IE_AE"]]
ie_ae["Z/#e"] = ie_ae["Z"] / ie_ae["#e"]
X = ie_ae[["Z/#e", "E_IQA_Intra(A)_b/Vee_ab(A,A)",
           "E_IQA_Intra(A)_a/Vee_ab(A,A)", "Ox", "N_alpha(A)", "N_total(A)", "VeeC(A,A)_a"]]
y = y.to_numpy()
X = X.to_numpy()


def linear_model(X, y, X_t, y_t_v):
    model = LinearRegression()
    model.fit(X, y)
    r_2, y_pre = model.score(X, y), model.predict(X)
    mae, max_ = mean_absolute_error(y, y_pre), max_error(y, y_pre)
    y_pre_t = model.predict(X_t)
    r_2_t = model.score(X_t, y_t_v)
    mae_t, max_t = mean_absolute_error(y_t_v, y_pre_t), max_error(y_t_v, y_pre_t)
    return [r_2, mae, max_, r_2_t, mae_t, max_t], [y, y_pre, y_t_v, y_pre_t]


fold = KFold(n_splits=10, shuffle=True, random_state=19)
scaler = StandardScaler(with_mean=True, with_std=True)
fold_n = 1
metrics = []
predicted = []
plt.rcParams['font.size'] = '30'

for train, validate in fold.split(X, y):
    x_tr, x_va, y_tr, y_va = scaler.fit_transform(X[train]), scaler.transform(
        X[validate]), y[train], y[validate]
    print("Evaluando fold #", fold_n)
    me, pre = linear_model(x_tr, y_tr, x_va, y_va)
    metrics.append(me), predicted.append(pre)
    plt.figure(figsize=(40, 20))
    plt.scatter(pre[0], pre[1], s=100, color="blue")
    plt.plot(pre[0], pre[0], color="red")
    plt.grid(True, "both", "both")
    plt.xlabel("Experimental (Hartree)")
    plt.ylabel("Predicted (Hartree)")
    plt.title("Train: fold " + str(fold_n) + " R2: " +
              f"{me[0]:.3f}" + " MAE: " + f"{me[1]:.3f}" + " Max: " + f"{me[2]:.3f}")
    plt.savefig("/mnt/c/Users/richi/Desktop/QTAIM/NPA/true_pre/Ml/" +
                str(fold_n) + "_train.jpg")
    plt.close()
    plt.figure(figsize=(40, 20))
    plt.scatter(pre[2], pre[3], s=100, color="blue")
    plt.plot(pre[2], pre[2], color="red")
    plt.xlabel("Experimental (Hartree)")
    plt.ylabel("Predicted (Hartree)")
    plt.grid(True, "both", "both")
    plt.title("Val: fold " + str(fold_n) + " R2: " + f"{me[3]:.3f}" +
              " MAE: " + f"{me[4]:.3f}" + " Max: " + f"{me[5]:.3f}")
    plt.savefig("/mnt/c/Users/richi/Desktop/QTAIM/NPA/true_pre/Ml/" +
                str(fold_n) + "_val.jpg")
    plt.close()
    fold_n += 1
metrics = np.array(metrics)
np.savetxt("Metrics_MLR.csv", metrics, delimiter=",")
print(metrics)
print(np.mean(metrics, axis=0))
