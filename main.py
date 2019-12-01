import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from core.preprocessing import Dataset
from core.model import Model
from core.config import cfg


# no_steps = [1028, 1203, 1249, 1290, 1315, 1409, 1494, 1495, 1017, 1041, 1157]
# pickle.load(open("./data/selected_patients.pickle", "rb"))
full_data_patients = [
    1014,
    1131,
    1159,
    1242,
    1259,
    1275,
    1348,
    1391,
    1408,
    1420,
    1431,
    1436,
    1443,
    1466,
    1474,
    1489,
    1346,
]
# Load and prepare data

# master_x_train = np.array()
# master_y_train = []
# master_x_val = []
# master_y_val = []
# , 'protein', 'fat', 'fiber', 'steps'
for i in range(full_data_patients.__len__()):
    # df = pd.read_csv(cfg.DATA.PATH)
    df = pd.read_csv("./data/" + str(full_data_patients[i]) + ".csv")
    x_train, y_train, x_val, y_val = (
        Dataset(df)
        .clean_df_2(columns_list=["timestamp", "glucose", "carbs", "steps"])
        .series_to_supervised()
    )
    print(x_train.shape)
    if i == 0:
        master_x_train = x_train
        master_y_train = y_train
        master_x_val = x_val
        master_y_val = y_val
    else:
        print(x_train.shape, master_x_train.shape)
        master_x_train = np.concatenate((master_x_train, x_train))
        master_y_train = np.concatenate((master_y_train, y_train))
        master_x_val = np.concatenate((master_x_val, x_val))
        master_y_val = np.concatenate((master_y_val, y_val))

print(
    master_x_train.shape, master_y_train.shape, master_x_val.shape, master_y_val.shape
)
# print(x_train[:5], y_train[:5], x_val[:5], y_val[:5])


# Load and train model
forecasting_model = Model().train(
    master_x_train,
    master_y_train,
    master_x_val,
    master_y_val,
    load_if_saved=False,
    name="multi_patient_carbs_steps_unscaled",
    train_again=True,
)

# Predict

# model = tf.keras.models.load_model('model.h5')
y_predicted = forecasting_model.predict(
    master_x_val[: 20 * cfg.TRAIN.BATCH_SIZE], master_y_val[: 20 * cfg.TRAIN.BATCH_SIZE]
)
# print(y_predicted)
