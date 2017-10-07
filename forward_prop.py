import numpy as np
import pandas as pd
import os

from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import time
import pickle

def pipeline_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
model2 = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
filelist = os.listdir('./imdb_crop_resized/imdb_crop_resized')
num_files = len(filelist)

# Next steps are very time-consuming

tic = time.time()
weight_matrix = np.ndarray(shape = (num_files,4096),dtype = np.float32)
weight_matrix2 = np.ndarray(shape = (num_files,4096),dtype = np.float32)
for i in range(num_files):
    img_path = './imdb_crop_resized/imdb_crop_resized/'+str(i)+'.jpg'
    processed = pipeline_preprocess(img_path)
    feature = model.predict(processed)
    feature2 = model.predict(processed)
    weight_matrix[i] = feature
    weight_matrix2[i] = feature2
    if i%10000 == 0: print(i)

toc = time.time()-tic

with open('weight_matrix.pkl','wb') as f:
    pickle.dump(weight_matrix,f)
with open('weight_matrix_fc1.pkl','wb') as f:
    pickle.dump(weight_matrix2,f)
