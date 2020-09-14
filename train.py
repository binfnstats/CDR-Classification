import glob
import pandas as pd
import re
import os
from pathlib import Path
from tqdm import tqdm
import requests
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import seaborn as sns
import math
from scipy.ndimage import zoom
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
import json
from resnet3dmulti import Resnet3DBuilder
from tensorflow.keras.regularizers import l2
from CdrDataset import CdrDataset
import sys

logdir = sys.argv[1]
input_shape = (193, 229, 193, 1)
output_shape = 5
train_batch_size = 4
test_batch_size = 4
ds = CdrDataset('data/combined/subjects.csv')

# Create a data generator for training and testing dataset.
def data_gen(df_dataset):
    for idx, row in df_dataset.iterrows():
        yield ds.get_sample(row, is_binary=False)

# Start cross fold validation.
df_train, df_independent_test = ds.get_train_test(['OASIS', 'ADNI1_Baseline_3T', 'ADNI1_Screening_1.5T', 'BMC'], include_augmented=True, test_size=0.2) 
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(df_train[['mri_path', 'age_year', 'sex']], df_train['cdr']):
    df_train, df_test = df_train[['mri_path', 'age_year', 'sex']].iloc[train_index], df_train[['mri_path', 'age_year', 'sex']].iloc[test_index]
    y_train, y_test = df_train[['cdr']].iloc[train_index], df_train[['cdr']].iloc[test_index]
    df_train['cdr'] = y_train
    df_test['cdr'] = y_test
    train_samples = df_train.shape[0]
    test_samples = df_test.shape[0]

    # Start distributed training.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Resnet3DBuilder.build_resnet_152(input_shape, output_shape)
        model.compile(optimizer=keras.optimizers.Adam(0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        keras.utils.plot_model(model, logdir + '/model.png', show_shapes=True, show_layer_names=False)            
        tb = keras.callbacks.TensorBoard(logdir, profile_batch=0)
        
        # The fix for a bug with tensorflow incorrectly logging redundant info.
        def noop():
            pass
        tb._enable_trace = noop
        
        callbacks = [
            tb,
            keras.callbacks.ModelCheckpoint(logdir + '/model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
            keras.callbacks.EarlyStopping(patience=10),
        ]
        dataset_train = tf.data.Dataset.from_generator(lambda: data_gen(df_train), output_types=({'img_input': tf.float32, 'age_input': tf.int32, 'gender_input': tf.int32}, tf.int32), output_shapes=({'img_input': input_shape, 'age_input': (1,), 'gender_input': (1,)}, (output_shape,))).batch(train_batch_size).repeat()
        dataset_test = tf.data.Dataset.from_generator(lambda: data_gen(df_test), output_types=({'img_input': tf.float32, 'age_input': tf.int32, 'gender_input': tf.int32}, tf.int32), output_shapes=({'img_input': input_shape, 'age_input': (1,), 'gender_input': (1,)}, (output_shape,))).batch(test_batch_size).repeat()
        history = model.fit(dataset_train, validation_data=dataset_test, steps_per_epoch=math.ceil(train_samples/train_batch_size), validation_steps=math.ceil(test_samples/test_batch_size), epochs=200, verbose=1, callbacks=callbacks, use_multiprocessing=True)
        return history
