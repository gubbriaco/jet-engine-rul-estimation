import enum
import os
import shutil
import re
import time
import numpy as np
import pandas as pd
import json
import joblib
from tqdm import tqdm
from typing import Callable, Dict, List, Union
import pickle
import gdown
from pprint import pprint

from matplotlib import pyplot as plt
import seaborn as sns
import hiplot as hip

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

import tensorflow
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, callbacks, losses
import keras_tuner as kt
# tf.debugging.set_log_device_placement(True)
devices = [device for device in tensorflow.config.list_physical_devices() if device.device_type == 'GPU']
if not devices:
    devices = tensorflow.config.list_physical_devices()[0]
print('devices: {}'.format(devices))

seed_random = 34
np.random.seed(seed_random)


in_dir = './input'
out_dir = './working'

in_dataset_name = 'CMAPSSData'
dataset_dir = os.path.join(in_dir, in_dataset_name)

dataset_id = 4
datasets_name = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_name = datasets_name[dataset_id-1]
dataset_path = os.path.join(dataset_dir, datasets_name[dataset_id-1])

'''DATASET URL (CMAPSS DATASET)'''
dataset_url = 'https://drive.google.com/file/d/1LU1DQuv7_CzBy2_Abgjg3HsvNDme361O/view?usp=drive_link'

'''PRE PROCESSED DATASET - /preprocessing/ directory'''
preprocessed_dataset_fd001_url = 'https://drive.google.com/file/d/1TnV9sAX2My-R3iSGamX3rVAezoBsGY_M/view?usp=drive_link'
preprocessed_dataset_fd002_url = 'https://drive.google.com/file/d/1pzFs93wfKkd4MZTATkQB-_oypFafq7iL/view?usp=drive_link'
preprocessed_dataset_fd003_url = 'https://drive.google.com/file/d/1zCT4nZMr0wnaGJZixayS_UJ1fh8ScxqP/view?usp=drive_link'
preprocessed_dataset_fd004_url = 'https://drive.google.com/file/d/1BGdAnxde8qRobh_nQUxTLPX2cDqBVSYD/view?usp=drive_link'
preprocessed_dataset_url = (
    preprocessed_dataset_fd001_url if dataset_id == 1 else
    preprocessed_dataset_fd002_url if dataset_id == 2 else
    preprocessed_dataset_fd003_url if dataset_id == 3 else
    preprocessed_dataset_fd004_url if dataset_id == 4 else
    None
)

models_name = 'models'
models_dir = os.path.join(in_dir, models_name)

'''TUNED CNN MODELS - /tuning/ directory'''
tuned_cnn_model_fd001_url = 'https://drive.google.com/file/d/107rKeqyMaDncCk6jpbyyCX86xzbMSiDU/view?usp=drive_link'
tuned_cnn_model_fd002_url = 'https://drive.google.com/file/d/1YpCySV1adGQKdIZbifyiolXsucTv9ZZt/view?usp=drive_link'
tuned_cnn_model_fd003_url = 'https://drive.google.com/file/d/10H7ofhmbeHkrTjZ8v-lGDd62xZkQ-Vjz/view?usp=drive_link'
tuned_cnn_model_fd004_url = 'https://drive.google.com/file/d/1L5AQKlno4RuPmusVC8qz9gmFjT6gsYNq/view?usp=drive_link'
tuned_cnn_model_url = (
    tuned_cnn_model_fd001_url if dataset_id == 1 else
    tuned_cnn_model_fd002_url if dataset_id == 2 else
    tuned_cnn_model_fd003_url if dataset_id == 3 else
    tuned_cnn_model_fd004_url if dataset_id == 4 else
    None
)

'''TRAINED TUNED CNN MODELS - /training/cnn/ directory'''
cnn_model_fd001_url = 'https://drive.google.com/file/d/1DF7tzAUChiIzXeU-OJ1oxL5zYIPFcuwb/view?usp=drive_link'
cnn_model_fd002_url = 'https://drive.google.com/file/d/1SVkcoowN2c81G_Z7Fz8Emu-31hpsJMcb/view?usp=drive_link'
cnn_model_fd003_url = 'https://drive.google.com/file/d/1cZuIOhbS6_p-VenunqBJ-Q7-RBjyy2ge/view?usp=drive_link'
cnn_model_fd004_url = 'https://drive.google.com/file/d/169jt7fdcTgzp6Oz1CE58wbo22-xa5fRI/view?usp=drive_link'
cnn_model_url = (
    cnn_model_fd001_url if dataset_id == 1 else
    cnn_model_fd002_url if dataset_id == 2 else
    cnn_model_fd003_url if dataset_id == 3 else
    cnn_model_fd004_url if dataset_id == 4 else
    None
)

'''TRAINED LSTM MODELS - /training/lstm/ directory'''
lstm_model_fd001_url = 'https://drive.google.com/file/d/1j9xSyJ8b-5Mu3aL2SJvMXg8jYRFwfLrS/view?usp=drive_link'
lstm_model_fd002_url = 'https://drive.google.com/file/d/1XSZQg2aH0Csd9kdrYqFJjpKPJGnzUMF4/view?usp=sharing'
lstm_model_fd003_url = ''
lstm_model_fd004_url = ''
lstm_model_url = (
    lstm_model_fd001_url if dataset_id == 1 else
    lstm_model_fd002_url if dataset_id == 2 else
    lstm_model_fd003_url if dataset_id == 3 else
    lstm_model_fd004_url if dataset_id == 4 else
    None
)

'''TRAINED GRU MODELS - /training/lstm/ directory'''
gru_model_fd001_url = 'https://drive.google.com/file/d/1c4E35MUR8BGLYiydqTZHPwU5ysQG6qTn/view?usp=sharing'
gru_model_fd002_url = ''
gru_model_fd003_url = ''
gru_model_fd004_url = ''
gru_model_url = (
    gru_model_fd001_url if dataset_id == 1 else
    gru_model_fd002_url if dataset_id == 2 else
    gru_model_fd003_url if dataset_id == 3 else
    gru_model_fd004_url if dataset_id == 4 else
    None
)

'''DATAFRAME COLUMNS TO DROP VIA FEATURE IMPORTANCE- /eda/ directory'''
columns_to_drop = (
    [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23] if dataset_id == 1 else 
    [0, 1, 2, 3, 4, 5, 9, 14, 20, 22, 23] if dataset_id == 2 else
    [0, 1, 2, 3, 4, 5, 9, 14, 20, 22, 23] if dataset_id == 3 else
    [0, 1, 2, 3, 4, 5, 9, 20, 22, 23] if dataset_id == 4 else
    None
)

'''TIME-SERIES WINDOW LENGTH - /preprocessing/ directory'''
window_length = (
    30 if dataset_id == 1 else 
    20 if dataset_id == 2 else
    30 if dataset_id == 3 else
    15 if dataset_id == 4 else
    None
)

'''TIME-SERIES UPPER RUL VALUE - /preprocessing/ directory'''
upper_rul = (
    125 if dataset_id == 1 else 
    150 if dataset_id == 2 else
    125 if dataset_id == 3 else
    150 if dataset_id == 4 else
    None
)

'''TIME-SERIES SHIFT VALUE - /preprocessing/ directory'''
shift = 1

'''TIME-SERIES NUMBER OF WINDOWS - /preprocessing/ directory'''
num_windows = 5


out_data_name = 'data'
out_data_dir = os.path.join(out_dir, out_data_name)
out_plots_name = 'plots'
out_plots_dir = os.path.join(out_dir, out_plots_name)
out_models_name = models_name
out_models_dir = os.path.join(out_dir, out_models_name)
out_tuning_name = 'tuning'
out_tuning_dir = os.path.join(out_dir, out_tuning_name)
out_evaluation_name = 'evaluation'
out_evaluation_dir = os.path.join(out_dir, out_evaluation_name)


'''SEPARATOR FOR READING DATA FROM FILES'''
pdsep = r'\s+'

'''VALIDATION DATASET SIZE VIA INITIAL TRAINING DATASET SIZE'''
val_size = 0.2

random_state_split = 83
