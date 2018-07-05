# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 23:51:01 2018

@author: Vijay
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

ip_folder = r"C:\Users\Vijay\Desktop\kaggle"
train = r"train.csv"
test = r"test.csv"

df_train = pd.read_csv(os.path.join(ip_folder, train))
df_test = pd.read_csv(os.path.join(ip_folder, test))

df_train_cp = df_train.copy(deep=True)
df_train = df_train_cp.copy(deep=True)

list_col_float = df_train.columns.tolist()
list_col_float.remove('ID')

scaler = StandardScaler()
scaler.fit(df_train[list_col_float])
arr_train_sc = scaler.transform(df_train[list_col_float])
df_train_sc = pd.DataFrame(data=arr_train_sc[0:, 0:])
df_train_sc.columns = list_col_float

list_x = list_col_float[:]
list_x.remove('target')
list_y = ['target']
sp_rat = 0.3

df_train_x, df_val_x, df_train_y, df_val_y = datasplit(df_train_sc, list_x, list_y, sp_rat)

arr_train_x = df_train_x.values
arr_val_x = df_val_x.values
arr_train_y = df_train_y.values
arr_val_y = df_val_y.values

batch_size = 128
epochs = 20
acti = 'relu'

model = Sequential()
model.add(Dense(4096, activation=acti, input_shape=(4991,)))
model.add(Dropout(0.2))
model.add(Dense(2048, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(1024, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(512, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(256, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(128, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(64, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(32, activation=acti))
model.add(Dropout(0.2))
model.add(Dense(1, activation=acti))

model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

history = model.fit(arr_train_x, arr_train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(arr_val_x, arr_val_y))

score = model.evaluate(arr_val_x, arr_val_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def datasplit(df_data, list_x, list_y, sp_rat):
    df_train_x, df_val_x, df_train_y, df_val_y = train_test_split(df_data[list_x], df_data[list_y],
                                                                  test_size=sp_rat, random_state=42)
    df_train_x.sort_index(inplace=True)
    df_val_x.sort_index(inplace=True)
    df_train_y.sort_index(inplace=True)
    df_val_y.sort_index(inplace=True)
    return df_train_x, df_val_x, df_train_y, df_val_y
