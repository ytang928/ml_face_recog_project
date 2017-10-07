import numpy as np
import pandas as pd
import os
import pickle


from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import time

base_model = VGG16(weights='imagenet')
fc2_parameter = base_model.layers[-2].get_weights()
with open('fc2_params.pkl','wb') as f:
    pickle.dump(fc2_parameter,f)