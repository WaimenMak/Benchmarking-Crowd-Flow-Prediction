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


    # Define the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    # params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse", "tree_method":"gpu_hist"}
    params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse"}
    # Fit the model to the training data
    model.set_params(**params_xgb)
    start_time = time.time()
    logger.info(f"training start:")
    model.fit(train_X, train_y, eval_set=eval_set, verbose=True)
    # model.fit(train_X, train_y)
    end_time = time.time()
    total_train_time = end_time - start_time


    # test_mse_loss = []
    # test_mask_rmse_loss = []
    # test_mask_mae_loss = []
    # test_mask_mape_loss = []
    # half_test_mask_rmse_loss = []
    # half_test_mask_mae_loss = []
    # half_test_mask_mape_loss = []
    # end_test_mask_rmse_loss = []
    # end_test_mask_mae_loss = []
    # end_test_mask_mape_loss = []
    if args.mode == "in-sample":
        model.save_model('./result/boost/xgb_regressor.model')
        loaded_model = xgb.Booster()
        loaded_model.load_model('./result/boost/xgb_regressor.model')
    else:
        model.save_model('./result/boost/xgb_regressor_ood.model')
        loaded_model = xgb.Booster()
        loaded_model.load_model('./result/boost/xgb_regressor_ood.model')

    test_ml(data, loaded_model, args, logger, total_train_time)
    # test_dataloader = data["test_loader"]
    #
    # for batch_idx, (input, org_target) in enumerate(test_dataloader.get_iterator()):
    #     for i in range(args.features): #3 feature num
    #         input[..., i] = data["scalers"][i].inverse_transform(input[..., i]) # turn to original data
    #
    #     input = input.reshape([-1, args.num_nodes * args.features * args.seq_len])
    #     # target = org_target.reshape([-1, args.num_nodes * args.features * args.seq_len])
    #     # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #     label = org_target
    #     for i in range(args.features):
    #         label[..., i] = data["scalers"][i].inverse_transform(label[..., i])   #normalize
    #
    #     output = loaded_model.predict(xgb.DMatrix(input))
    #     output = output.reshape([args.batch_size, args.seq_len, args.num_nodes, args.features])
    #
    #
    #     test_rmse = [np.sum(np.sqrt(np.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, axis=(1,2)))) for step_t in range(12)]
    #     test_rmse = sum(test_rmse) / len(test_rmse) / args.batch_size
    #
    #
    #     test_mse_loss.append(test_rmse.item())
    #     test_mask_rmse_loss.append(masked_rmse_np(output, label)) # avg
    #     half_test_mask_rmse_loss.append(masked_rmse_np(output[:,5,:,:], label[:,5,:,:])) # half
    #     end_test_mask_rmse_loss.append(masked_rmse_np(output[:,11,:,:], label[:,11,:,:])) # end
    #     test_mask_mae_loss.append(masked_mae_np(output, label))
    #     half_test_mask_mae_loss.append(masked_mae_np(output[:,5,:,:], label[:,5,:,:]))
    #     end_test_mask_mae_loss.append(masked_mae_np(output[:,11,:,:], label[:,11,:,:]))
    #     for i in range(args.features):
    #         output[..., i] = data["scalers"][i].transform(output[..., i])   #normalize
    #         label[..., i] = data["scalers"][i].transform(label[..., i])   #normalize
    #     test_mask_mape_loss.append(masked_mape_np(output, label))
    #     half_test_mask_mape_loss.append(masked_mape_np(output[:,5,:,:], label[:,5,:,:]))
    #     end_test_mask_mape_loss.append(masked_mape_np(output[:,11,:,:], label[:,11,:,:]))
    #
    #
    # # test_mse_loss = test_mse_loss / test_iters
    # test_mse_loss = np.mean(test_mse_loss)
    #
    # test_mask_rmse_loss = np.mean(test_mask_rmse_loss)
    # test_mask_mape_loss = np.mean(test_mask_mape_loss)
    # test_mask_mae_loss = np.mean(test_mask_mae_loss)
    # half_test_mask_rmse_loss = np.mean(half_test_mask_rmse_loss)
    # half_test_mask_mape_loss = np.mean(half_test_mask_mape_loss)
    # half_test_mask_mae_loss = np.mean(half_test_mask_mae_loss)
    # end_test_mask_rmse_loss = np.mean(end_test_mask_rmse_loss)
    # end_test_mask_mape_loss = np.mean(end_test_mask_mape_loss)
    # end_test_mask_mae_loss = np.mean(end_test_mask_mae_loss)
    # logger.info(f"model: {args.filename}, testing method: {args.mode}, test_RMSE: {test_mse_loss:.4f}")
    # logger.info(f"avg_test_MASK_RMSE: {test_mask_rmse_loss:.4f}, avg_test_MASK_MAE: {test_mask_mae_loss:.4f}, avg_test_MASK_MAPE: {test_mask_mape_loss:.4f}")
    # logger.info(f"half_test_MASK_RMSE:{half_test_mask_rmse_loss:.4f}, half_test_MASK_MAE: {half_test_mask_mae_loss:.4f}, half_test_MASK_MAPE: {half_test_mask_mape_loss:.4f}")
    # logger.info(f"end_test_MASK_RMSE:{end_test_mask_rmse_loss:.4f}, end_test_MASK_MAE: {end_test_mask_mae_loss:.4f}, end_test_MASK_MAPE: {end_test_mask_mape_loss:.4f}, Time: {total_train_time:.4f}")

