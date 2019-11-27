import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from datetime import datetime

from .config import cfg

FMT = '%Y-%m-%d %H:%M:%S'

class Dataset():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.params = cfg.DATA

    def clean_df(self): 
        self.data = self.dataframe.fillna(0).drop(columns = ['heart rate']).sort_values(by=['Time'])[self.dataframe.apply(lambda x: datetime.strptime(x['Time'],FMT) > datetime.strptime("2019-02-26", "%Y-%m-%d"), axis=1)]
        self.values = self.data.values
        print(self.values.shape)
        return self

    def clean_df_2(self, columns_list=['timestamp', 'glucose']):
        self.data = self.dataframe.fillna(0)[columns_list].sort_values(by=['timestamp'])
        self.values = self.data.values
        print(self.values.shape)
        if len(columns_list)-1 != self.params.NUM_FEATURES :
            print("WARNING!!! Number of features do not match number of columns given!")
        return self

    def __rolling_window(self, y_index=1):
        k1 = self.params.INPUT_TIMESTEPS
        k2 = self.params.OUTPUT_TIMESTEPS
        n = len(self.values)
        
        self.X = []
        self.Y = []

        for i in range(n-k2-k1):
            self.X.append(self.values[i:i+k1, y_index:])
            self.Y.append(self.values[i+k1:i+k1+k2, y_index])

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        return self.X, self.Y

    def series_to_supervised(self, y_index=1):
        self.__rolling_window(y_index)
        split = int(self.params.VALIDATION_SPLIT*len(self.X))

        if split%cfg.TRAIN.BATCH_SIZE != 0:
            split = split - (split%cfg.TRAIN.BATCH_SIZE)
        split2 = len(self.X) - split
        
        if split2%cfg.TRAIN.BATCH_SIZE != 0:
            split2 = split2 - (split2%cfg.TRAIN.BATCH_SIZE)
        x_train, y_train = self.X[:split], self.Y[:split]
        x_val, y_val = self.X[split:split+split2], self.Y[split:split+split2]

        return x_train, y_train, x_val, y_val

    def plot_variables(self, groups):
        i = 1
        # plot each required column
        plt.figure(figsize=(10, 10))
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(self.values[:, group])
            plt.title(self.data.columns[group], y=0.5, loc='right')
            i += 1
        plt.show()
