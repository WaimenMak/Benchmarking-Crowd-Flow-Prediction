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
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
import pickle
from lib.train_test import test_ml

# np.random.seed(42)
class VAR():
    def __init__(self, inpt_dim, step):
        self.A = None # the dimention A is related to the lags and input_dim, [lags * input_dim + 1, input_dim], where input_dim is the number of varibles, +1 is the columns with 1
        self.features = inpt_dim  # dimention k, k variable, 3 or 105
        self.lags = step

    def fit(self, x_train, y_train): # x train [N, seq * features]
        X = np.concatenate([x_train, np.ones([x_train.shape[0], 1])], axis=1)
        L = X.shape[1]
        X = X[:, (L-1) - self.lags*self.features:]
        # Compute the coefficients using least squares
        self.A = np.linalg.lstsq(X, y_train, rcond=None)[0]
        # return A

    def predict(self, x_test):
        '''
        A: [seq len + 1, features], x test: [N, seq_len * features], here seq_len = lags
        x test is already been processed with the required input time steps
        '''
        if self.A is None:
            raise Exception("Matrix attribute is not initialized.")

        x_test = np.concatenate([x_test, np.ones([x_test.shape[0], 1])], axis=1) # x_test [N, seq_len * features + 1]
        # L = x_test.shape[1]
        multi_steps = []
        # x_test = x_test[:, (L-1) - self.lags*self.features:] # extract lags of features of the original testing data, if lags=12, there's nothing changed
        for step in range(12):
            x = x_test
            y = np.dot(x, self.A)
            x_temp = x_test[:, self.features:-1].copy() # start index shoud be 3, end should be 1, because last column is 1 should not be changed.
            x_test[:, -1-self.features: -1] = y # the last three columns are replaced by preicition y
            x_test[:, : -1-self.features] = x_temp # the previous columns are replaced by the original columns
            multi_steps.append(y)

        predictions = np.stack(multi_steps, axis=1)
        return predictions # [N, steps, features]

    def load(self, path):
        self.A = np.load(path)

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    file_handler = logging.FileHandler("./result/train VAR_v2.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # parser = argparse.ArgumentParser()
    # total_train_time = 0
    # args.step = 3
    args.batch_size = 64
    args.seq_len = 12
    args.num_nodes = 35
    args.features = 3
    args.inpt_dim = args.features
    # parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    # args = parser.parse_args()
    if args.mode == "in-sample":
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)
    else:
        raise ValueError

    # Define the LRoost model
    models = [VAR(args.inpt_dim, args.step) for _ in range(args.num_nodes)]
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    for j, model in enumerate(models):
        eval_set = [(data["x_val"][:,:,j,:].reshape([-1, args.seq_len*args.features]), data["y_val"][:,0,j,:])]
        """
        here we use seq_len, rather than step is because inside the model it choose the lag by itself. 
        For other model we select the lag for them by indicating the step in advance and feed it to model.
        """
        model.fit(data["x_train"][:,:,j,:].reshape([-1, args.seq_len*args.features]), data["y_train"][:,0,j,:]) # the ground trith only contains step 0

    end_time = time.time()
    total_train_time = end_time - start_time

    loaded_models = []
    if args.mode == "in-sample":
        for j, model in enumerate(models):
            np.save(f'./result/VAR/VAR{j}.npy', model.A)
            model.load(f'./result/VAR/VAR{j}.npy')
            loaded_models.append(model)
    elif args.mode == "ood":
        loaded_models = []
        for j, model in enumerate(models):
            np.save(f'./result/VAR/VAR_ood{j}.npy', model.A)
            model.load(f'./result/VAR/VAR_ood{j}.npy')
            loaded_models.append(model)

    test_ml(data, loaded_models, args, logger, total_train_time)

if __name__ == '__main__':

    batch_size = 308
    seq_len = 12
    num_nodes = 35
    features = 3
    # Example usage
    outputs = []
    loaded_models = []
    train_steps = 3
    mode = "insample"
    source = np.random.rand(batch_size, seq_len, num_nodes, features)
    # source_xgb, target_xgb = sliding_win(vis_data)
    for j in range(num_nodes):
        loaded_model = VAR(inpt_dim=3, step=3)
        # loaded_model.load_model(f'./result/boost/xgb_regressor{j}.model')
        if mode == "insample":
            loaded_model.load(f'../result/VAR/VAR{j}.npy')
        else:
            loaded_model.load(f'../result/VAR/VAR_ood{j}.npy')
        output = loaded_model.predict(source[:, -train_steps:,j,:].reshape([-1, train_steps*features]))
        output = output.reshape([-1, seq_len, source.shape[3]])
        outputs.append(output)

