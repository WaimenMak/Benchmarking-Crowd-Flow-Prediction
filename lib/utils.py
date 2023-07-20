import logging

import numpy
import numpy as np
import pandas as pd
import os
import pickle
import scipy.sparse as sp
import sys

from numpy import ndarray
from scipy.sparse import linalg
import dgl
import math
import torch
import torch.nn as nn

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

def init_seq2seq(module):  #@save
    """Initialize weights for Seq2Seq."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    # adj_mx = sp.coo_matrix(adj_mx)   # original version, adj is not sparse matrix
    d = np.array(adj_mx.sum(1))
    # d_inv = np.power(d, -1).flatten()
    d_inv = 1/d
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(np.squeeze(d_inv))
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    # data{x and dataloader}
    data = {}
    scalers = []
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'].astype("float64")
        data['y_' + category] = cat_data['y'].astype("float64")

    for i in range(3):   # 3 features for every node
        if i != 2:
            scaler = StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std())
            scalers.append(scaler)
        else:
            scaler = StandardScaler(mean=(scalers[0].mean + scalers[1].mean),
                                    std=np.sqrt(scalers[0].std**2+scalers[1].std**2))
            scalers.append(scaler)
        # Data format
        for category in ['train', 'val', 'test']:
            # temp = scalers[i].transform(data['x_' + category][..., i])
            data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
            data['y_' + category][..., i] = scalers[i].transform(data['y_' + category][..., i])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scalers'] = scalers

    return data

def load_dataset_rnn(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    # data{x and dataloader}
    data = {}
    scalers = []
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        seq_len = cat_data['x'].shape[1]
        feature_size = cat_data['x'].shape[3]
        data['x_' + category] = cat_data['x'].transpose(0, 2, 1, 3).reshape(-1, seq_len, feature_size)
        data['y_' + category] = cat_data['y'].transpose(0, 2, 1, 3).reshape(-1, seq_len, feature_size)

    for i in range(3):   # 3 features for every node
        if i != 2:
            scaler = StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std())
            scalers.append(scaler)
        else:
            scaler = StandardScaler(mean=(scalers[0].mean + scalers[1].mean),
                                    std=np.sqrt(scalers[0].std**2+scalers[1].std**2))
            scalers.append(scaler)
        # Data format
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
            data['y_' + category][..., i] = scalers[i].transform(data['y_' + category][..., i])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scalers'] = scalers

    return data



def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

class EarlyStopper:
    def __init__(self, tolerance=1, min_delta=0):
        self.tolerance = tolerance
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_delta = min_delta
        self.last_val_loss = np.inf

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # if val_loss < self.last_val_loss:
            #     self.counter = 0
            if self.counter >= self.tolerance:
                return True
        self.last_val_loss = val_loss
        return False


def build_graph():
    # data
    # coord = np.array(
    #     [[0, 1], [1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [5, 8], [6, 8], [7, 8],
    #      [8, 9], [9, 10], [10, 11],
    #      [9, 11], [9, 12], [10, 12], [11, 12], [12, 13], [13, 14], [13, 16], [14, 16], [14, 15], [15, 16], [13, 15],
    #      [16, 17], [17, 18],
    #      [2, 19], [2, 20], [19, 20], [20, 21], [3, 21], [21, 22], [3, 22], [6, 23], [23, 24],
    #      [6, 24], [24, 25], [7, 25], [25, 26], [7, 26], [10, 27], [10, 28], [27, 28], [28, 29], [11, 29], [29, 30],
    #      [11, 30], [14, 31], [31, 32], [14, 32],
    #      [15, 33], [33, 34], [32, 33], [15, 34]])
    coord = np.array(
    [[1, 3], [3, 4], [4, 5], [4, 6], [4, 7], [5, 33], [5, 7], [6, 7], [5, 6], [33, 29],
     [33, 32], [32, 29], [35, 32], [35, 29], [35, 6], [7, 9], [9, 10], [9, 11], [9, 12],
     [10, 12], [10, 11], [11, 12], [10, 28], [28, 26], [28, 31], [26, 31], [34, 26], [34, 31],
     [34, 11], [12, 14], [14, 15], [14, 16], [14, 17],[15, 16], [15, 17], [16, 17], [15, 25],
     [25, 27], [24, 27],[24, 25],[30, 27],[30, 24], [30, 16], [17, 19], [19, 22], [19, 21],
     [19, 20], [22, 20], [22, 21], [21, 20], [22, 18], [18, 8], [18, 13], [8, 13], [23, 8],
     [23, 13], [23, 21], [20, 2]])
    coord = coord - 1
    src = coord[:, 0]
    dst = coord[:, 1]

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    # print(u,v)
    return dgl.graph((u, v))

def _compute_sampling_threshold(global_step, k):
    """
    Computes the sampling probability for scheduled sampling using inverse sigmoid.
    :param global_step:
    :param k:
    :return:
    """
    return k / (k + math.exp(global_step / k))



def train(model):
    import time
    training_iter_time = 0
    total_train_time = 0
    G = build_graph()
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="coo")
    batch_size = 64
    enc_input_dim = 3  # encoder network input size, can be 1 or 3
    dec_input_dim = 3  # decoder input
    max_diffusion_step = 2
    num_nodes = 35
    # num_rnn_layers = 2
    num_heads = 5
    hidden_dim = 64
    seq_len = 12
    output_dim = 3
    device = "cpu"

    max_grad_norm = 5
    cl_decay_steps = 2000
    data = load_dataset("../dataset", batch_size=64, test_batch_size=64)
    data_loader = data["train_loader"]
    val_dataloader = data["val_loader"]
    test_dataloader = data["test_loader"]
    # model = GAT(adj_mat, seq_len, enc_input_dim, hidden_dim, output_dim, num_nodes, batch_size, num_heads)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True)
    num_samples = data["x_train"].shape[0]
    val_samples = data["x_val"].shape[0]
    test_samples = data["x_test"].shape[0]
    train_iters = math.ceil(num_samples / batch_size)
    val_iters = math.ceil(val_samples / batch_size)
    test_iters = math.ceil(test_samples / batch_size)
    early_stopper = EarlyStopper(tolerance=15, min_delta=0.01)
    # training_iter_time = num_samples / batch_size
    # len_epoch = math.ceil(num_samples / batch_size)
    len_epoch = 100  #500
    train_list, val_list, test_list = [], [], []
    model.to(device)
    for epoch in range(1, len_epoch):
        model.train()
        train_rmse_loss = 0
        # training_time += train_epoch_time
        '''
        code from dcrnn_trainer.py, _train_epoch()
        '''
        start_time = time.time()  # record the start time of each epoch
        total_loss = 0
        # total_metrics = np.zeros(len(metrics))
        for batch_idx, (data, target) in enumerate(data_loader.get_iterator()):
            data = torch.FloatTensor(data)
            # data = data[..., :1]
            target = torch.FloatTensor(target)
            label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            #data [bc, seq, node, feature]
            # for i in range(num_nodes):
            output = model(data.permute(0, 2, 1, 3).reshape([batch_size, num_nodes, -1])) # [bc, node, output_dim]


            # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
            # loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
            loss_sup_seq = [torch.sum((output - label[:, step_t, :, :]) ** 2) for step_t in range(1)]  #training loss function
            train_rmse = [torch.sum(torch.sqrt(torch.sum((output - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)] # training metric for each step

            loss = sum(loss_sup_seq) / len(loss_sup_seq) / batch_size
            train_rmse = sum(train_rmse) / len(train_rmse) / batch_size
            # loss = loss(output.cpu(), label)  # loss is self-defined, need cpu input
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            training_iter_time = time.time() - start_time
            total_train_time += training_iter_time

            # writer.set_step((epoch - 1) * len_epoch + batch_idx)
            # writer.add_scalar('loss', loss.item())
            # total_loss += loss.item()
            # train_mse_loss += loss.item()
            train_rmse_loss += train_rmse.item()  #metric  sum of each node and each iteration
            # total_metrics += _eval_metrics(output.detach().numpy(), label.numpy()

            # if batch_idx % log_step == 0:
            #     logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         _progress(batch_idx),
            #         loss.item())

            if batch_idx == len_epoch:
                break

        train_rmse_loss = train_rmse_loss / train_iters
        print(f"Epoch: {epoch}, train_RMSE: {train_rmse_loss}")



        # validation
        if epoch % 5 == 0 and epoch != 0 :
            val_mse_loss = 0
            # validation
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_dataloader.get_iterator()):
                    data = torch.FloatTensor(data)
                    # data = data[..., :1]
                    target = torch.FloatTensor(target)
                    label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
                    data, target = data.to(device), target.to(device)

                    # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0) # 0:teacher forcing rate
                    output = model(data.permute(0, 2, 1, 3).reshape([batch_size, num_nodes, -1]))
                    # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
                    # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
                    #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)

                    val_rmse = [torch.sum(torch.sqrt(torch.sum((output - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]   # 12 graphs
                    val_rmse = sum(val_rmse) / len(val_rmse) / batch_size

                    val_mse_loss += val_rmse.item()



            val_mse_loss = val_mse_loss / val_iters

            print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
            train_list.append(train_rmse_loss)
            val_list.append(val_mse_loss)
            np.save("../result/gat_train_loss.npy", train_list)
            np.save("../result/gat_val_loss.npy", val_list)

            torch.save(model.state_dict(), "./result/gat.pt")
            if early_stopper.early_stop(val_mse_loss):
                break
                # pass
    # Testing
    test_mse_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader.get_iterator()):
            data = torch.FloatTensor(data)
            # data = data[..., :1]
            target = torch.FloatTensor(target)
            label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(device), target.to(device)

            # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0)
            # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
            #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)
            output = model(data.permute(0, 2, 1, 3).reshape([batch_size, num_nodes, -1]))
            # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
            test_rmse = [torch.sum(torch.sqrt(torch.sum((output - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]
            test_rmse = sum(test_rmse) / len(test_rmse) / batch_size

            test_mse_loss += test_rmse.item()

    test_mse_loss = test_mse_loss / test_iters
    print(f"test_MSE: {test_mse_loss}, Time: {total_train_time}")

def sliding_win(data: numpy.ndarray) -> tuple:
    """
    slicing the data by sliding window
    """
    x_offsets = np.sort(
            # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
            np.concatenate((np.arange(-11, 1, 1),))
        )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0]- abs(max(y_offsets)))

    # max_t = abs(N - abs(max(y_offsets)))  # Exclusive
    x, y = [], []
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def plot_multiple_figures(pred:list, target:numpy, models:list, step:int, node:int, feature:int, path:str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    n = 0
    for i in range(3):
        for j in range(3):
            axes[i, j].plot(np.arange(target.shape[0]), pred[n][:, step, node, feature], label=f'Prediction')
            axes[i, j].plot(np.arange(target.shape[0]), target[n][:, step, node, feature], label=f'Ground Truth')
            axes[i, j].legend()
            axes[i, j].set_title(f'Model {models[n]}')
            n += 1


    fig.suptitle(f'Predictions and Targets on Step {step+1}, Node {node+1}')
    fig.savefig(f'{path}/insample.pdf')
    plt.show()

def gen_data_dict(df_dict:dict):
    data_dict = {}
    for k in df_dict.keys():
        mat_list = []
        for col in df_dict[k].columns:
            mat_list.append(np.array(df_dict[k][col].values.tolist()))
        data_dict[k] = np.stack(mat_list, axis=1)

def process_sensor_data(parent_dir, df_dict):
    for subdir, dirs, files in os.walk(parent_dir):
        # Initialize an empty list to hold the dataframes for this subdirectory
        df_list = []
        # Loop through each file in the subdirectory
        files.sort()
        for filename in files:
            # Check if the file is a Excel file
            if filename.endswith('.xlsx'):
                df = pd.read_excel(subdir + '/' + filename, header=None, index_col=0)
                df = df.T
                temp = df.columns
                for i in range(len(temp)):
                    if pd.isna(temp[i]):
                        df.iloc[:, i] = df.iloc[:, i-2] + df.iloc[:, i-1]
                        df = df.rename(columns={df.columns[i]: 'Sum'})

                # add a new column that contains the sum of the last two columns
                df.insert(df.shape[1], '', df.iloc[:, -2] + df.iloc[:, -1])
                # rename the last column to "Sum"
                df = df.rename(columns={df.columns[-1]: "Sum"})
                for i in range(len(temp)):
                    if df.columns[i] == "Station_hall_layer" or df.columns[i] == "Platform_layer":
                        sub_df = df.iloc[:,i+1:i+5]
                        flow_data = np.stack([sub_df["Left to Right"].values, sub_df["Right to Left"].values, sub_df["Sum"].values], axis=1).astype("int")
                        df_list.append(pd.Series([row for row in flow_data], name = "sensor_" + df.iloc[0, i].split(" ")[1]))

            if df_list:
                df_concatenated = pd.concat(df_list, axis=1)
                # print(subdir)
                sorted_cols = sorted(df_concatenated.columns, key=lambda x: int(x.split('_')[1])) # sort the column name
                df_concatenated = df_concatenated.reindex(columns=sorted_cols)
                # Add the concatenated dataframe to the dictionary with the subdirectory name as the key
                df_dict[subdir] = df_concatenated

    return df_dict