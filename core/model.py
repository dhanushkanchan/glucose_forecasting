import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d as gfilter

from .config import cfg


class Model:
    def __init__(self):
        # Is this really required??
        self.model_params = cfg.TRAIN
        self.data_params = cfg.DATA

    def __model(self):
        _inputs = tf.keras.layers.Input(
            batch_shape=(
                self.model_params.BATCH_SIZE,
                self.data_params.INPUT_TIMESTEPS,
                self.data_params.NUM_FEATURES,
            )
        )
        lstm = tf.keras.layers.LSTM(
            self.model_params.LSTM_NEURONS, stateful=self.model_params.STATEFUL
        )(_inputs)
        _outputs = tf.keras.layers.Dense(
            self.data_params.OUTPUT_TIMESTEPS, activation="linear"
        )(lstm)
        model = tf.keras.Model(inputs=_inputs, outputs=_outputs)
        model.compile(
            loss=self.model_params.LOSS,
            optimizer=self.model_params.OPTIMIZER,
            metrics=self.model_params.METRICS,
        )
        return model

    def __model2(self):
        # Some output shape issue between LSTM and Time distributed layers
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                self.model_params.LSTM_NEURONS,
                batch_input_shape=(
                    self.model_params.BATCH_SIZE,
                    self.data_params.INPUT_TIMESTEPS,
                    self.data_params.NUM_FEATURES,
                ),
                return_sequences=True,
                stateful=True,
            )
        )
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.data_params.OUTPUT_TIMESTEPS)
            )
        )
        model.add(tf.keras.layers.Activation("sigmoid"))
        model.compile(
            loss=self.model_params.LOSS,
            optimizer=self.model_params.OPTIMIZER,
            metrics=self.model_params.METRICS,
        )
        return model

    def plot_train_history(self, loss, val_loss, title):
        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title(title)
        plt.legend()

        plt.show()

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        load_if_saved=True,
        name="model",
        train_again=True,
    ):
        if load_if_saved:
            self.temp_model = tf.keras.models.load_model(name + ".h5")

        else:
            self.temp_model = self.__model()
            # checkpoint = ModelCheckpoint('../checkpoints/forecasting_model.h5', monitor='val_loss',save_best_only=True, mode='min')
        if train_again or not load_if_saved:
            callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
            print("NUM EPOCHS : ", self.model_params.EPOCHS)
            self._loss = []
            self._val_loss = []
            for i in range(self.model_params.EPOCHS):
                print(i)
                history = self.temp_model.fit(
                    x_train,
                    y_train,
                    epochs=1,
                    batch_size=self.model_params.BATCH_SIZE,
                    verbose=1,
                    shuffle=False,
                    validation_data=(x_val, y_val),
                )
                self._loss.append(history.history["loss"])
                self._val_loss.append(history.history["val_loss"])
                self.temp_model.reset_states()
            self.plot_train_history(
                self._loss, self._val_loss, "Training and validation loss"
            )
            tf.keras.models.save_model(self.temp_model, name + ".h5")

        return self

    def predict(self, x_test, y_test):
        # TODO for multi-input multi-output
        y_predicted = self.temp_model.predict(
            x_test, batch_size=self.model_params.BATCH_SIZE
        )
        # _y = [y[0] for y in y_test]
        # _ypred = [y[0] for y in y_predicted]
        n = len(y_test)
        # fig = go.Figure()
        # for i in range(n):
        #     fig.add_trace(go.Scatter(x=list(range(i, i+self.data_params.OUTPUT_TIMESTEPS)), y=y_predicted[i], name="Predicted", showlegend=False, line_color='red'))
        #     fig.add_trace(go.Scatter(x=list(range(i, i+self.data_params.OUTPUT_TIMESTEPS)), y=y_test[i], name="Ground Truth", showlegend=False, line_color='blue'))
        # fig.show()

        fig2 = go.Figure()
        mins = []
        maxs = []
        # _n = len(np.unique(i for i in range(j, j+self.data_params.OUTPUT_TIMESTEPS) for j in range(n)))
        # for i in range(n):
        #     for j in range(i, i+self.data_params.OUTPUT_TIMESTEPS):

        fig2.add_trace(
            go.Scatter(
                x=list(range(len(y_predicted))),
                y=gfilter([y[0] for y in y_predicted], sigma=2),
                line_color="rgba(147, 112, 219, 0.2)",
            )
        )
        for i in range(1, len(y_predicted[0])):
            fig2.add_trace(
                go.Scatter(
                    x=list(range(i, len(y_predicted) + i)),
                    y=gfilter([y[i] for y in y_predicted], sigma=2),
                    fill="tonexty",
                    line_color="rgba(147, 112, 219, 0.2)",
                )
            )
        for i in range(n):
            fig2.add_trace(
                go.Scatter(
                    x=list(range(i, i + self.data_params.OUTPUT_TIMESTEPS)),
                    y=y_test[i],
                    name="Ground Truth",
                    showlegend=False,
                    line_color="red",
                )
            )

        # fig2.add_trace(go.Scatter)

        fig2.show()

        return y_predicted
