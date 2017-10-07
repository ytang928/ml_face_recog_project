from scipy.io import loadmat
from datetime import *
import datetime
import time
import numpy as np
import pandas as pd
import os
import itertools
import shutil
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
# import cv2
import PIL
import pickle
from keras.models import Sequential
from keras.layers import Dense

with open('weight_matrix_all_selected.pkl','rb') as f:
    weight_matrix_all_selected = pickle.load(f)
with open('weight_matrix_all_fc1_selected.pkl','rb') as f:
    weight_matrix_all_fc1_selected = pickle.load(f)

with open('sex.pkl','rb') as f:
    sex = pickle.load(f)
with open('age.pkl','rb') as f:
    age = pickle.load(f)
with open('fc2_params.pkl','rb') as f:
    fc2_params = pickle.load(f)
sex = sex.astype(np.float32)
age = age.astype(np.float32)

x = weight_matrix_all_fc1_selected
y_sex = sex.ravel()
y_age = age.ravel()

x_train = x[:180000]
x_test = x[180000:]
y_age_train = y_age[:180000]
y_age_test = y_age[180000:]
y_sex_train = y_sex[:180000]
y_sex_test = y_sex[180000:]

model_sex = Sequential()
model_sex.add(Dense(4096, input_dim=4096, activation='relu',name = 'fc2',kernel_regularizer = regularizers.l2(0.01)))
model_sex.add(Dropout(0.5))
model_sex.get_layer('fc2').set_weights(fc2_params)
model_sex.add(Dense(1, activation='sigmoid', name = 'fc3'))
model_sex.get_layer('fc3').set_weights(model_prep.get_layer('fc3_pretrain').get_weights())
model_sex.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
for i in range(50):
    print(i+1)
    model_sex.fit(x_train, y_sex_train, epochs=10, batch_size=4096, verbose=True)
    score_model_sex = model_sex.evaluate(x_test, y_sex_test, verbose=0)
    model_sex.save('model_sex_'+str(i+1))
    print(score_model_sex)

model_age = Sequential()
model_age.add(Dense(4096, input_dim=4096, activation='relu',name = 'fc2',kernel_regularizer = regularizers.l2(0.01)))
model_age.get_layer('fc2').set_weights(fc2_params)
model_age.add(Dense(1, input_dim=4096,name = 'fc3'))
model_age.get_layer('fc3').set_weights(model3.layers[0].get_weights())
model_age.compile(loss='mean_absolute_error', optimizer='adam')
for i in range(50):
    print(i+1)
    model_age.fit(x_train, y_age_train, epochs=10, batch_size=4096, verbose=True)
    score_model_age = model_age.evaluate(x_test, y_age_test, verbose=True)
    model_age.save('model_age'+str(i)+'.mdl')
    print(score_model_age)




