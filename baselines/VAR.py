# -*- coding: utf-8 -*-
# @Time    : 17/05/2023 15:26
# @Author  : mmai
# @FileName: VAR
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 16/05/2023 10:37
# @Author  : mmai
# @FileName: VAR_v2
# @Software: PyCharm

import numpy as np
import argparse
from lib.utils import load_dataset
import time
import logging
from baselines.VAR_v2 import VAR
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
import pickle
from lib.train_test import test_ml


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    file_handler = logging.FileHandler("./result/train VAR_v2.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # parser = argparse.ArgumentParser()
    # total_train_time = 0
    args.step = 3  # input seq length, empriacally set to 3 has the best performance
    args.batch_size = 64 # don't change
    args.seq_len = 12  # don't change
    args.num_nodes = 35 # don't change
    args.features = 3  # don't change
    args.inpt_dim = args.features * args.num_nodes
    # parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    # args = parser.parse_args()
    if args.mode == "in-sample":
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)
    else:
        raise ValueError

    train_X = data['x_train'].astype("float64").reshape([-1, args.seq_len*args.features * args.num_nodes])
    train_y = data['y_train'].astype("float64")
    train_y = train_y[:, :1, :,:].reshape([-1, 1 * 3 * 35]) # variable numbers become 3 * 35 rather than 3 anymore

    val_X = data['x_val'].astype("float64").reshape([-1, args.seq_len*args.features * args.num_nodes])
    val_y = data['y_val'].astype("float64")
    val_y = val_y[:, :1, :,:].reshape([-1, 1 * 3 * 35]) # :1 means only one step ground truth

    eval_set = [(val_X, val_y)]
    # Define the LRoost model
    model = VAR(args.inpt_dim, args.step)
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    model.fit(train_X, train_y)
    end_time = time.time()
    total_train_time = end_time - start_time


    if args.mode == "in-sample":
        np.save(f'./result/VAR/VAR.npy', model.A)
        model.load(f'./result/VAR/VAR.npy')

    elif args.mode == "ood":
        np.save(f'./result/VAR/VAR_ood.npy', model.A)
        model.load(f'./result/VAR/VAR_ood.npy')


    test_ml(data, model, args, logger, total_train_time)

if __name__ == '__main__':

    batch_size = 64
    seq_len = 12
    num_nodes = 35
    features = 3
    # Example usage
    x_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # N = 2, T = 4
    y_train = np.array([[10, 20], [50, 60]])  # N = 2, M = 2

    p = 2  # Lag order
    model = VAR(args)
    data = load_dataset("../dataset", batch_size=64, test_batch_size=64)
    # Fit the VAR model
    j = 1
    A = model.fit(data["x_train"][:,:,j,:].reshape([-1, seq_len*features]), data["y_train"][:,0, j,:]) # step = 0, one step prediction for training

    x_test = np.array([[2, 3, 4, 5], [6, 7, 8, 9]])  # N = 2, T = 4

    # Perform predictions
    predictions = model.predict(A, data["x_test"][:,:,j,:].reshape([-1, seq_len*features]))

    # Print the predictions
    print("Predictions:")
    print(predictions)
