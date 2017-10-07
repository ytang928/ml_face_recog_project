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
import cv2
import PIL
import pickle

def calculate_age(x):
    count = 0
    year_take = int(x[0].split('/')[1].split('_')[-1].split('.')[0])
    birth = int(x[0].split('/')[1].split('_')[2].split('-')[0])
    if birth == 0: 
        birth = 1970
    diff = year_take - birth
    return diff

mat = loadmat('./Downloads/imdb_crop/imdb.mat')

age = np.array(list(map(calculate_age,mat['imdb'][0][0][2][0]))).reshape(460723,1)
sex = mat['imdb'][0][0][3][0].reshape(460723,1)
filename = np.array(list(map(lambda x:x[0].split('/')[1], mat['imdb'][0][0][2][0]))).reshape(460723,1)
year_taken = mat['imdb'][0][0][1][0].reshape(460723,1)
first_face = mat['imdb'][0][0][6][0].reshape(460723,1)
second_face = mat['imdb'][0][0][7][0] .reshape(460723,1)

data_desc = pd.DataFrame(np.concatenate([filename,sex,age,year_taken,first_face,second_face],axis = 1),columns = ['filename','sex','age','year_taken','first_face','second_face']).set_index('filename').sort_index()
data_desc['original_name'] = data_desc.index
data_desc.index = pd.RangeIndex(0,460723)
data_desc_processed = data_desc.sort_values(by = 'first_face',ascending=False)
data_desc_processed['age'] = data_desc_processed['age'].astype(np.float)
data_desc_processed['sex'] = data_desc_processed['sex'].astype(np.float)
data_desc_processed['year_taken'] = data_desc_processed['year_taken'].astype(np.int)
data_desc_processed['first_face'] = data_desc_processed['first_face'].astype(np.float)
data_desc_processed['second_face'] = data_desc_processed['second_face'].astype(np.float)

size = pd.Series(index = pd.RangeIndex(0,460723))
for i in range(460723):
    size[i] = os.path.getsize('./Downloads/imdb_crop_processed/'+str(i)+'.jpg')
#     if not i%10000:print(i)

size_new = pd.Series(index = pd.RangeIndex(0,460723))
for i in range(460723):
    size_new[i] = os.path.getsize('./Downloads/imdb_crop_resized/'+str(i)+'.jpg')
    
data_desc_processed['size'] = size
data_desc_processed['size_new'] = size_new\

first_score_processed = data_desc_processed['first_face'].replace([np.inf, -np.inf], np.nan).dropna()
first_score_processed_02quantile = first_score_processed.quantile(0.2)
first_score_processed = first_score_processed[first_score_processed>first_score_processed_02quantile]
firsts_second = data_desc_processed.loc[first_score_processed.index,'second_face']
second_score_processed = data_desc_processed['second_face']
second_score_processed_08quantile = second_score_processed.quantile(0.8)
firsts_second = firsts_second[np.bitwise_not(firsts_second>second_score_processed_08quantile)]
data_desc_processed = data_desc_processed.loc[firsts_second.index]
age_noerror = data_desc_processed.loc[data_desc_processed['age']<90]
data_desc_processed = data_desc_processed.loc[age_noerror.index]
age_noerror = data_desc_processed.loc[data_desc_processed['age']>10]
data_desc_processed = data_desc_processed.loc[age_noerror.index]

original_threshold = data_desc_processed['size'].sort_values().quantile(0.2)
new_threshold = data_desc_processed['size_new'].sort_values().quantile(0.2)
size_OK = data_desc_processed['size']>original_threshold
size_new_OK = data_desc_processed['size_new']>new_threshold
data_desc_processed = data_desc_processed.loc[size_OK]
data_desc_processed = data_desc_processed.loc[size_new_OK]

sex_OK = np.bitwise_not(pd.isnull(data_desc_processed['sex']))
data_desc_processed = data_desc_processed.loc[sex_OK]

data_desc_processed.sort_index(ascending=True,inplace=True)
data_desc_processed['new_index'] = range(data_desc_processed.shape[0])

selected_photo_index = data_desc_processed.index.tolist()

with open('model.pkl','wb') as f:
    pickle.dump(data_desc_processed,f)

with open('weight_matrix_all.pkl','rb') as f:
    weight_matrix = pickle.load(f)
weight_matrix_selected = weight_matrix[selected_photo_index]
with open('weight_matrix_all_selected.pkl','wb') as f:
    pickle.dump(weight_matrix_selected,f)

with open('weight_matrix_fc1_all.pkl','rb') as f:
    weight_matrix_fc1 = pickle.load(f)
weight_matrix_fc1_selected = weight_matrix_fc1[selected_photo_index]
with open('weight_matrix_all_fc1_selected.pkl','wb') as f:
    pickle.dump(weight_matrix_fc1_selected,f)

