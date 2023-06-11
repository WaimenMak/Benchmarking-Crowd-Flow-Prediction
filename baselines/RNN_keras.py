# -*- coding: utf-8 -*-
# @Time    : 02/05/2023 22:20
# @Author  : mmai
# @FileName: RNN_keras
# @Software: PyCharm


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from lib.train_test import test_ml
import numpy as np
import argparse
from lib.utils import load_dataset
import time
import logging
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler("./result/train RNN keras.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# np.random.seed(42)

def main(args):
    import math
    # parser = argparse.ArgumentParser()
    # args.step = 3
    args.batch_size = 64
    args.seq_len = 12
    args.num_nodes = 35
    args.features = 3
    args.n_outputs = args.seq_len * args.features * args.num_nodes
    # parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    # args = parser.parse_args()
    if args.mode == "in-sample":
        # train = np.load("./dataset/train.npz")
        # val = np.load("./dataset/val.npz")
        # test = np.load("./dataset/test.npz")
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        # train = np.load("./ood_dataset/train.npz")
        # val = np.load("./ood_dataset/val.npz")
        # test = np.load("./ood_dataset/test.npz")
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)
    else:
        raise ValueError


    # test_samples = data["x_test"].shape[0]
    # test_iters = math.ceil(test_samples / batch_size)

    # train_X = train['x'].astype("float64").reshape([-1, seq_len, features * num_nodes])
    # train_y = train['y'].astype("float64")
    # train_X = data['x_train'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.seq_len, args.features * args.num_nodes])
    train_X = data['x_train'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step, args.features * args.num_nodes])
    train_y = data['y_train'].astype("float64")
    train_y = train_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    # test_X = test['x'].astype("float64").reshape([-1, 12 * 3 * 35])
    # test_y = test['y'].astype("float64")
    # test_y = test_y[:, :step, :,:].reshape([-1, step * 3 * 35])

    # val_X = val['x'].astype("float64").reshape([-1, seq_len, features * num_nodes])
    # val_y = val['y'].astype("float64")
    # val_y = val_y[:, :step, :,:].reshape([-1, step * 3 * 35])

    val_X = data['x_val'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step, args.features * args.num_nodes])
    val_y = data['y_val'].astype("float64")
    # val_y = val_y[:, :args.step, :,:].reshape([-1, args.step * 3 * 35])
    val_y = val_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    # eval_set = [(val_X, val_y)]
    # n_samples = 100
    # n_features = 12
    # Split the data into training and testing sets
    # train_size = int(n_samples * 0.8)


    # Define the XGBoost model
    # params_lr = {"normalize": True}
    model = Sequential()
    # model.add(SimpleRNN(64, input_shape=(seq_len, features)))
    model.add(GRU(64, activation='relu', input_shape=(args.step, args.features * args.num_nodes)))
    model.add(Dense(args.n_outputs))
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    model.compile(loss='mse', optimizer='adam')
    # define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    # Fit the model to the training data
    model.fit(train_X, train_y, epochs=200, batch_size=64, validation_data=(val_X, val_y), callbacks=[early_stop])
    end_time = time.time()
    total_train_time = end_time - start_time

    if args.mode == "in-sample":
        model.save('./result/RNN keras.h5')
        loaded_model = load_model('./result/RNN keras.h5')

    else:
        model.save('./result/linear/RNN keras ood.h5')
        loaded_model = load_model('./result/linear/RNN keras ood.h5')

    test_ml(data, loaded_model, args, logger, total_train_time)
