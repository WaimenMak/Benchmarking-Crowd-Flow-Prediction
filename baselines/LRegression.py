import numpy as np
from lib.utils import load_dataset
from lib.train_test import test_ml
import time
import logging
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
from sklearn.linear_model import LinearRegression
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler("./result/train LR.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# np.random.seed(42)

def main(args):

    # args.step = 12
    args.batch_size = 64
    args.seq_len = 12
    args.num_nodes = 35
    args.features = 3

    if args.mode == "in-sample":
        # train = np.load("../dataset/train.npz")
        # val = np.load("../dataset/val.npz")
        # test = np.load("../dataset/test.npz")
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        # train = np.load("../ood_dataset/train.npz")
        # val = np.load("../ood_dataset/val.npz")
        # test = np.load("../ood_dataset/test.npz")
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)
    else:
        raise ValueError

    # different from VAR, here x should be processed by args.step
    train_X = data['x_train'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step*args.features * args.num_nodes])
    train_y = data['y_train'].astype("float64")
    train_y = train_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    # test_X = test['x'].astype("float64").reshape([-1, 12 * 3 * 35])
    # test_y = test['y'].astype("float64")
    # test_y = test_y[:, :step, :,:].reshape([-1, step * 3 * 35])

    # val_X = val['x'].astype("float64").reshape([-1, seq_len, features * num_nodes])
    # val_y = val['y'].astype("float64")
    # val_y = val_y[:, :step, :,:].reshape([-1, step * 3 * 35])

    val_X = data['x_val'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step*args.features * args.num_nodes])
    val_y = data['y_val'].astype("float64")
    val_y = val_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    eval_set = [(val_X, val_y)]
    # n_samples = 100
    # n_features = 12
    # Split the data into training and testing sets
    # train_size = int(n_samples * 0.8)


    # params_lr = {"normalize": True}
    model = LinearRegression()
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    model.fit(train_X, train_y) # annotate this line when testing
    end_time = time.time()
    total_train_time = end_time - start_time


    if args.mode == "in-sample":
        pickle.dump(model, open('./result/linear/LR_regressor.sav', 'wb')) # annotate this line when testing
        loaded_model = pickle.load(open('./result/linear/LR_regressor.sav', 'rb'))

    else:
        pickle.dump(model, open('./result/linear/LR_regressor_ood.sav', 'wb')) # # annotate this line when testing
        loaded_model = pickle.load(open('./result/linear/LR_regressor_ood.sav', 'rb'))

    test_ml(data, loaded_model, args, logger, total_train_time)

