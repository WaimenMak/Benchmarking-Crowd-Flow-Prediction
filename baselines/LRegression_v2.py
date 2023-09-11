# -*- coding: utf-8 -*-
# @Time    : 21/04/2023 16:52
# @Author  : mmai
# @FileName: LRegression_v2.py
# @Software: PyCharm

import numpy as np
import argparse
from lib.utils import load_dataset
import time
import logging
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
from sklearn.linear_model import LinearRegression
import pickle
from lib.train_test import test_ml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler("./result/train LR_v2.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# np.random.seed(42)

def main(args):
    # parser = argparse.ArgumentParser()
    # total_train_time = 0
    # step = 12
    args.batch_size = 64
    args.seq_len = 12
    args.num_nodes = 35
    args.features = 3
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

    # Define the LR model
    models = [LinearRegression() for i in range(args.num_nodes)]
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    for j, model in enumerate(models):
        """
        different from VAR, here the input data x should be processed with the seq lenth, e.g. args.seq_len --> args.step
        """
        eval_set = [(data["x_val"][:,-args.step:,j,:].reshape([-1, args.step*args.features]), data["y_val"][:,:,j,:].reshape([-1, args.seq_len*args.features]))]
        model.fit(data["x_train"][:,-args.step:,j,:].reshape([-1, args.step*args.features]), data["y_train"][:,:,j,:].reshape([-1, args.seq_len*args.features]))

    end_time = time.time()
    total_train_time = end_time - start_time


    loaded_models = []
    if args.mode == "in-sample":
        for j, model in enumerate(models):
            pickle.dump(model, open(f'./result/linear/LR_regressor{j}.sav', 'wb')) # annotate this line when testing
            loaded_model = pickle.load(open(f'./result/linear/LR_regressor{j}.sav', 'rb'))
            loaded_models.append(loaded_model)
    elif args.mode == "ood":
        loaded_models = []
        for j, model in enumerate(models):
            pickle.dump(model, open(f'./result/linear/LR_regressor_ood{j}.sav', 'wb')) # annotate this line when testing
            loaded_model = pickle.load(open(f'./result/linear/LR_regressor_ood{j}.sav', 'rb'))
            loaded_models.append(loaded_model)

    test_ml(data, loaded_models, args, logger, total_train_time)
