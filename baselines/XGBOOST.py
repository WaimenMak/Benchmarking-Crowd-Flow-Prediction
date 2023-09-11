import xgboost as xgb
import numpy as np
from lib.train_test import test_ml
from lib.utils import load_dataset
import time
import logging
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler("./result/train XGBOOST.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# np.random.seed(42)

def main(args):
    # parser = argparse.ArgumentParser()
    # args.step = 12
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

    # train_X = train['x'].astype("float64").reshape([-1, seq_len, features * num_nodes])
    # train_y = train['y'].astype("float64")
    # train_y = train_y[:, :step, :,:].reshape([-1, step * 3 * 35])
    # train_X = data['x_train'].astype("float64").reshape([-1, args.seq_len * args.features * args.num_nodes])
    train_X = data['x_train'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step*args.features * args.num_nodes])
    train_y = data['y_train'].astype("float64")
    # train_y = train_y[:, :args.step, :,:].reshape([-1, args.step * 3 * 35])
    train_y = train_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    val_X = data['x_val'].astype("float64")[:, -args.step:,:,:].reshape([-1, args.step*args.features * args.num_nodes])
    val_y = data['y_val'].astype("float64")
    val_y = val_y[:, :args.seq_len, :,:].reshape([-1, args.seq_len * 3 * 35])

    eval_set = [(val_X, val_y)]
    # n_samples = 100
    # n_features = 12
    # Split the data into training and testing sets
    # train_size = int(n_samples * 0.8)


    # Define the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    # params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse", "tree_method":"gpu_hist"}
    params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse"}
    # Fit the model to the training data
    model.set_params(**params_xgb)
    start_time = time.time()
    logger.info(f"training start:")
    # model.fit(train_X, train_y, eval_set=eval_set, verbose=True)   # annotate for testing

    end_time = time.time()
    total_train_time = end_time - start_time

    if args.mode == "in-sample":
        # model.save_model('./result/boost/xgb_regressor.model') # annotate for testing
        loaded_model = xgb.Booster()
        loaded_model.load_model('./result/boost/xgb_regressor.model')
    else:
        # model.save_model('./result/boost/xgb_regressor_ood.model') # testing # annotate for testing
        loaded_model = xgb.Booster()
        loaded_model.load_model('./result/boost/xgb_regressor_ood.model')

    test_ml(data, loaded_model, args, logger, total_train_time)

