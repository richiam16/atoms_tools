from keras.models import Sequential
from keras.layers import Dense, PReLU
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_absolute_error, max_error, r2_score
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')

# Importing the data
IE = pd.read_csv("IE_AE_DNN_2.csv")
y = IE["IE_AE"]
IE["Z/#e"] = IE["Z"] / IE["#e"]
IE = IE[["Z/#e", "E_IQA_Intra(A)_b/Vee_ab(A,A)",
         "E_IQA_Intra(A)_a/Vee_ab(A,A)", "Ox", "N_alpha(A)", "N_total(A)", "VeeC(A,A)_a"]]
X = IE.to_numpy()
y = y.to_numpy()

# The neural network parameters
epoch = 1000
learning_rate = 0.009
decay = learning_rate / epoch
adam = Adam(learning_rate=learning_rate, decay=decay)

# A neural network with 2 layers
def nn_model():
    model = Sequential()
    model.add(Dense(80, activation="relu", input_shape=(X.shape[1],)))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=adam, loss="mse", metrics=["mae"])
    return model

# Making the crossvalidation of the system
loss_h = []
predicted = []
save_dir = '/home/richiam/old_scripts/ie_deep_learning/outpout/model_'
fold = KFold(n_splits=10, shuffle=True, random_state=19)
scaler = StandardScaler(with_mean=True, with_std=True)
norm = Normalizer()
val_mae = []
metrics = []
plt.rcParams['font.size'] = '30'
fold_n = 1
for train, validate in fold.split(X, y):
    model = nn_model()
    x_tr, x_va, y_tr, y_va = scaler.fit_transform(X[train]), scaler.transform(
        X[validate]), y[train], y[validate]
    print("Evaluando fold #", fold_n)
    # log_dir = log_dir + str(fold_n)
    # tensorboard_ = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = ModelCheckpoint(save_dir + str(fold_n) + ".h5",
                                 monitor="val_mae", verbose=0, save_best_only=True, mode="min")
    csv = CSVLogger(save_dir + str(fold_n) + ".csv", separator=",", append=False)
    detente = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="min")
    history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), callbacks=[
                        checkpoint, detente], batch_size=10, epochs=epoch, verbose=1)
    model.load_weights(save_dir + str(fold_n) + ".h5")
    results = model.evaluate(x_va, y_va)
    results = dict(zip(model.metrics_names, results))
    y_pre_tr = model.predict(x_tr)
    y_pre_va = model.predict(x_va)
    metrics.append([r2_score(y_tr, y_pre_tr), mean_absolute_error(y_tr, y_pre_tr), max_error(
        y_tr, y_pre_tr), r2_score(y_va, y_pre_va), mean_absolute_error(y_va, y_pre_va), max_error(y_va, y_pre_va)])
    val_mae.append(results["mae"])
    loss_h.append(history.history["val_mae"])
    predicted.append([y_tr, y_pre_tr, y_va, y_pre_va])
    plt.figure(figsize=(40, 20))
    plt.scatter(y_tr, y_pre_tr, s=100, color="blue")
    plt.plot(y_tr, y_tr, color="red")
    plt.grid(True, "both", "both")
    plt.xlabel("Experimental (Hartree)")
    plt.ylabel("Predicted (Hartree)")
    plt.title("Train: fold " + str(fold_n) + " R2: " +
              f"{metrics[fold_n-1][0]:.3f}" + " MAE: " + f"{metrics[fold_n-1][1]:.3f}" + " Max: " + f"{metrics[fold_n-1][2]:.3f}")
    plt.savefig("/home/richiam/old_scripts/ie_deep_learning/outpout" +
                str(fold_n) + "_train.jpg")
    plt.close()
    plt.figure(figsize=(40, 20))
    plt.scatter(y_va, y_pre_va, s=100, color="blue")
    plt.plot(y_va, y_va, color="red")
    plt.xlabel("Experimental (Hartree)")
    plt.ylabel("Predicted (Hartree)")
    plt.grid(True, "both", "both")
    plt.title("Val: fold " + str(fold_n) + " R2: " + f"{metrics[fold_n-1][3]:.3f}" +
              " MAE: " + f"{metrics[fold_n-1][4]:.3f}" + " Max: " + f"{metrics[fold_n-1][5]:.3f}")
    plt.savefig("/home/richiam/old_scripts/ie_deep_learning/outpout" +
                str(fold_n) + "_val.jpg")
    plt.close()
    fold_n += 1
print(val_mae)
print("Val_mae_prom:", (sum(val_mae) / len(val_mae)) * 27.2114)
metrics = np.array(metrics)
prom = np.mean(metrics, axis=0)
np.savetxt("Metrics_DNN.csv", metrics, delimiter=",")
print(metrics)
print(prom)
