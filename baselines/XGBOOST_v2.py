import xgboost as xgb
import numpy as np
import argparse
from lib.utils import load_dataset
from lib.train_test import test_ml
import time
import logging
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler("./result/train XGBOOST_v2.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# np.random.seed(42)

def main(args):
    import math
    # parser = argparse.ArgumentParser()
    # total_train_time = 0
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


    # test_samples = data["x_test"].shape[0]
    # test_iters = math.ceil(test_samples / args.batch_size)


    # n_samples = 100
    # n_features = 12
    # Split the data into training and testing sets
    # train_size = int(n_samples * 0.8)


    # Define the XGBoost model
    models = [xgb.XGBRegressor(objective='reg:squarederror') for i in range(args.num_nodes)]
    params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse"}
    # Fit the model to the training data
    start_time = time.time()
    logger.info(f"training start:")
    for j, model in enumerate(models):
        model.set_params(**params_xgb)
        """
        different from VAR, here the input data x should be processed with the seq lenth, e.g. args.seq_len --> args.step
        """
        eval_set = [(data["x_val"][:,-args.step:,j,:].reshape([-1, args.step*args.features]),
                     data["y_val"][:,:,j,:].reshape([-1, args.seq_len*args.features]))]
        # model.fit(data["x_train"][:,-args.step:,j,:].reshape([-1, args.step*args.features]),   # test
        #           data["y_train"][:,:,j,:].reshape([-1, args.seq_len*args.features]), eval_set=eval_set, verbose=True)

    end_time = time.time()
    total_train_time = end_time - start_time

    if args.mode == "in-sample":
        loaded_models = []
        for j, model in enumerate(models):
            # model.save_model(f'./result/boost/xgb_regressor{j}.model') # test
            loaded_model = xgb.Booster()
            loaded_model.load_model(f'./result/boost/xgb_regressor{j}.model')
            loaded_models.append(loaded_model)
    elif args.mode == "ood":
        loaded_models = []
        for j, model in enumerate(models):
            # model.save_model(f'./result/boost/xgb_regressor_ood{j}.model') # test
            loaded_model = xgb.Booster()
            loaded_model.load_model(f'./result/boost/xgb_regressor_ood{j}.model')
            loaded_models.append(loaded_model)

    test_ml(data, loaded_models, args, logger, total_train_time)
    # test_dataloader = data["test_loader"]
    #
    # for batch_idx, (input, org_target) in enumerate(test_dataloader.get_iterator()):
    #     for i in range(features): #3 feature num
    #         input[..., i] = data["scalers"][i].inverse_transform(input[..., i]) # turn to original data
    #
    #     # input = input.reshape([-1, num_nodes * features * seq_len])
    #     # target = org_target.reshape([-1, num_nodes * features * seq_len])
    #     # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #     label = org_target
    #     for i in range(features):
    #         label[..., i] = data["scalers"][i].inverse_transform(label[..., i])   #normalize
    #
    #     outputs = []
    #     for j, loaded_model in enumerate(loaded_models):
    #         output = loaded_model.predict(xgb.DMatrix(input[:,:,j,:].reshape([-1, seq_len*features])))
    #         output = output.reshape([batch_size, seq_len, features])
    #         outputs.append(output)
    #
    #     output = np.stack(outputs, axis=2)
    #     test_rmse = [np.sum(np.sqrt(np.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, axis=(1,2)))) for step_t in range(12)]
    #     test_rmse = sum(test_rmse) / len(test_rmse) / batch_size
    #
    #     test_mse_loss += test_rmse.item()
    #     test_mask_rmse_loss.append(masked_rmse_np(output, label)) # avg
    #     half_test_mask_rmse_loss.append(masked_rmse_np(output[:,5,:,:], label[:,5,:,:])) # half
    #     end_test_mask_rmse_loss.append(masked_rmse_np(output[:,11,:,:], label[:,11,:,:])) # end
    #     test_mask_mae_loss.append(masked_mae_np(output, label))
    #     half_test_mask_mae_loss.append(masked_mae_np(output[:,5,:,:], label[:,5,:,:]))
    #     end_test_mask_mae_loss.append(masked_mae_np(output[:,11,:,:], label[:,11,:,:]))
    #     for i in range(features):
    #         output[..., i] = data["scalers"][i].transform(output[..., i])   #normalize
    #         label[..., i] = data["scalers"][i].transform(label[..., i])   #normalize
    #     test_mask_mape_loss.append(masked_mape_np(output, label))
    #     half_test_mask_mape_loss.append(masked_mape_np(output[:,5,:,:], label[:,5,:,:]))
    #     end_test_mask_mape_loss.append(masked_mape_np(output[:,11,:,:], label[:,11,:,:]))
    #
    #
    # test_mse_loss = test_mse_loss / test_iters
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

