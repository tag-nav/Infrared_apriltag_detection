import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
import random as rn
from os import mkdir
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import sys
sys.path.insert(0, "../")

seed = 0
os.environ["PYTHONHASHSEED"] = "0"

np.random.seed(seed)
rn.seed(seed)
tf.random.set_seed(seed)

from deeparuco.impl.architectures import custom_decoder
from deeparuco.impl.datagen import custom_decoder_gen


# Control paramters
batch_size = 32
epochs = 100
patience = 20
reduce_after = 10

# Model
model = custom_decoder()
model.summary
model.compile(loss="mae", optimizer="adam")


# Load dataset
train_src_dir = '../dataset/inpaint/crops_orig/train'
valid_src_dir = '../dataset/inpaint/crops_orig/val'
train_csv     = '../dataset/inpaint/crops_orig/train128.csv'
valid_csv     = '../dataset/inpaint/crops_orig/val128.csv'
train_df = pd.read_csv(train_csv)
valid_df = pd.read_csv(valid_csv)

train_generator = custom_decoder_gen(train_df, train_src_dir, batch_size, False, True)
valid_generator = custom_decoder_gen(valid_df, valid_src_dir, batch_size, False, True)

# Callbacks

stop = EarlyStopping(
    monitor="val_loss",
    patience=patience,
    verbose=True,
    restore_best_weights=True,
    min_delta=1e-4,
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=reduce_after, factor=0.5)

run_name = 'decoder'

if not exists("../models/inpaint"):
    mkdir("../models/inpaint")

    
model_name = "simple_decoder"
csv_logger = CSVLogger(f"../models/inpaint/loss_{run_name}.csv")


model.fit(
    train_generator,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[stop, reduce_lr, csv_logger],
    verbose=True,
)
model.save(f"../models/inpaint/{run_name}.keras")