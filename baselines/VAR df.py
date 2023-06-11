# -*- coding: utf-8 -*-
"""
Created on 09/01/2023 11:54

@Author: mmai
@FileName: VAR df.py
@Software: PyCharm
"""
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd
import numpy as np
import os
from lib.utils import StandardScaler
from lib.metric import masked_rmse_np, masked_mae_np, masked_mape_np

def read_data():
    features = 3
    num_nodes = 35
    window = 12
    file_dir = "./sensor data"  # file directory
    all_csv_list = os.listdir(file_dir)  # get csv list
    all_csv_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    single_data_frame = pd.read_csv(os.path.join(file_dir, 'sensor_0.csv'), sep="\t", index_col="Index")
    N = len(single_data_frame) # 1200
    data = np.zeros([N, num_nodes, features])
    for (i, single_csv) in enumerate(all_csv_list):
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv), sep="\t", index_col="Index")
        data[:,i,:] = single_data_frame[['Left to Right', 'Right to Left', 'Sum']].values

    return data    #[N, node, feat]

def var_predict(df, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):  # code from metric.py from DCRNN.
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    # data = scaler.transform(df_train.values)
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test

def var_predict_ood(df, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):  # code from metric.py from DCRNN.
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    # data = scaler.transform(df_train.values)
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test

def eval_var(df, sc, n_lags=3):
    print(sc)
    # train test
    n_forwards = [1, 3, 6, 12]

    y_predicts, y_test = var_predict(df, n_forwards=n_forwards, n_lags=n_lags,
                                     test_ratio=0.2)

    # logger.info('VAR (lag=%d)' % n_lags)
    # logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    results_half = np.zeros([3])
    results_end = np.zeros([3])

    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)  #y_predicts is a list
        mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        # line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        # logger.info(line)
        if horizon == 6:
            results_half[0] = rmse
            results_half[1] = mae
            results_half[2] = mape
        if horizon == 12:
            results_end[0] = rmse
            results_end[1] = mae
            results_end[2] = mape

        print('VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae))
    # rmse = masked_rmse_np(preds=y_predicts.values, labels=y_test.values, null_val=0)  #y_predicts is a list
    # mape = masked_mape_np(preds=y_predicts.values, labels=y_test.values, null_val=0)
    # mae = masked_mae_np(preds=y_predicts.values, labels=y_test.values, null_val=0)
    # print('VAR\tavg\t%.2f\t%.2f\t%.2f' % (rmse, mape * 100, mae))
    return results_half, results_end

def load_dataset(dataset_dir):
    # data{x and dataloader}
    import json
    with open(dataset_dir, 'r') as f:
        df_dict = json.load(f)

    train, test = [], []
    train_sc = ['./sc sensor/sc4_2', './sc sensor/sc13', './sc sensor/sc1', './sc sensor/sc6',
                './sc sensor/sc7_2', './sc sensor/sc7', './sc sensor/sc3_2', './sc sensor/sc6_2',
                './sc sensor/sc2_2', './sc sensor/sc13_2', './sc sensor/sc5', './sc sensor/sc2',
                './sc sensor/sc5_2', './sc sensor/sc3', './sc sensor/sc4']
    # convert the values from list to numpy array
    for key in df_dict:
        df_dict[key] = np.array(df_dict[key]).reshape(-1, 3 * 35) # feature * nodes
        if key in train_sc:
            train.append(df_dict[key])
        else:
            test.append(df_dict[key])

    train = np.stack(train, axis=0)
    test = np.column_stack(test)

    return train, test

if __name__ == '__main__':
    import json
    with open("../dataset/df_dict.json", 'r') as f:
        var_df_dict = json.load(f)

    # convert the JSON strings back to Pandas DataFrame objects
    for key in var_df_dict:
        var_df_dict[key] = pd.read_json(var_df_dict[key])

    train_sc = ['./sc sensor/sc4_2', './sc sensor/sc13', './sc sensor/sc1', './sc sensor/sc6',
                './sc sensor/sc7_2', './sc sensor/sc7', './sc sensor/sc3_2', './sc sensor/sc6_2',
                './sc sensor/sc2_2', './sc sensor/sc13_2', './sc sensor/sc5', './sc sensor/sc2',
                './sc sensor/sc5_2', './sc sensor/sc3', './sc sensor/sc4']

    test_sc = []
    for sc in var_df_dict.keys():
        if sc not in train_sc:
            test_sc.append(sc)

    res1, res2 = zip(*[eval_var(var_df_dict[sc], sc, n_lags=3) for sc in test_sc])

    res1, res2 = np.stack(res1, axis=0), np.stack(res2, axis=0)
    print('VAR\thalf\t%.2f\t%.2f\t%.2f' % (np.mean(res1[:,0]), np.mean(res1[:,1]), np.mean(res1[:,2] * 100)))
    print('VAR\tend\t%.2f\t%.2f\t%.2f' % (np.mean(res2[:,0]), np.mean(res2[:,1]), np.mean(res2[:,2] * 100)))
