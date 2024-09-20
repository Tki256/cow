import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import argparse
parser = argparse.ArgumentParser()    # 2. パーサを作る

# --- ハイパーパラメータの指定 ---
parser.add_argument("--lookback", type=int, default=500)   
parser.add_argument("--pred_length", type=int, default=50)   
parser.add_argument("--step", type=int, default=1)   
parser.add_argument("--delay", type=int, default=1)   
parser.add_argument("--batch_size", type=int, default=200)
   
# ------------------------


def generator(data, lookback, delay, pred_length, min_index, max_index, shuffle=False,
              batch_size=100, step=1):
    if max_index is None:
        max_index = len(data) - delay - pred_length - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index,
                                    size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                               lookback//step,
                               data.shape[-1]))

        targets = np.zeros((len(rows), pred_length))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay : rows[j] + delay + pred_length].flatten()

        yield samples, targets

# --- ハイパーパラメータ ---
lookback = 500
pred_length = 50 # 2.0s * 500Hz
step = 1
delay = 1
batch_size = 200
# ------------------------


df = pd.read_csv("data/Cow_Heartbeat_Separation.csv")
TIMEEVENT = df["TimeEvent"]
TIMEEVENT.dropna(inplace=True)
print(df.info())

normal_cycle = df["EMG"][10000:].values
normal_cycle = normal_cycle.reshape(-1, 1)

# 訓練ジェネレータ
train_gen = generator(normal_cycle,
                     lookback=lookback,
                     pred_length=pred_length,
                     delay=delay,
                     min_index=0,
                     max_index=200000,
                     shuffle=True,
                     step=step,
                     batch_size=batch_size)

val_gen = generator(normal_cycle,
                   lookback=lookback,
                   pred_length=pred_length,
                   delay=delay,
                   min_index=200001,
                   max_index=290000,
                   step=step,
                   batch_size=batch_size)


# 検証データセット全体を調べるためにval_genから抽出する時間刻みの数
val_steps = (290000 - 200001 -lookback) // batch_size




wandb.init(project="cow_HeartRate_and_Breath",
           name="LSTM")

# モデルの構築
model = Sequential()
model.add(layers.LSTM(64, return_sequences = True, input_shape=(None,normal_cycle.shape[-1])))
model.add(layers.LSTM(32))
model.add(layers.Dense(pred_length))

model.compile(optimizer=RMSprop(), loss="mse")

history = model.fit(train_gen,
                    steps_per_epoch=200,
                    epochs=1000,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    callbacks=[WandbMetricsLogger(),
                               WandbModelCheckpoint("models/LSTM.keras")])

wandb.finish()

