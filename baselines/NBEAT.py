# -*- coding: utf-8 -*-
# @Time    : 22/10/2024 21:41
# @Author  : mmai
# @FileName: NBEAT
# @Software: PyCharm
# -*- coding: utf-8 -*-
"""
Created on 04/12/2022 16:44

@Author: mmai
@FileName: RNN.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
from nbeats_pytorch.model import NBeatsNet
from lib.train_test import Trainer
from lib.utils import EarlyStopper, load_dataset
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
import random
import argparse
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
device = "cpu"
logger.info(f"Using device: {device}")

class NBeatsModel(nn.Module):
    def __init__(self, args, input_dim, output_dim, num_blocks=1, hidden_dim=64):
        super(NBeatsModel, self).__init__()
        self.model = NBeatsNet(
            device=device,
            stack_types=[NBeatsNet.GENERIC_BLOCK] * num_blocks,
            forecast_length=output_dim,
            backcast_length=input_dim,
            hidden_layer_units=hidden_dim,
            share_weights_in_stack=False,
            nb_blocks_per_stack=1,
            thetas_dim=(7, 8, 3),
        )
        self.args = args

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        backcast, forecast = self.model(x)
        forecast = forecast.reshape(forecast.size(0), self.args.seq_len, self.args.num_nodes, self.args.features)
        return forecast

def main(args):
    args.batch_size = 64
    args.features = 3      # actual features
    args.num_nodes = 35
    args.max_grad_norm = 5
    args.seq_len = 12
    args.output_dim = 3
    args.enc_input_dim = args.features * args.num_nodes * args.seq_len
    args.output_dim = args.features * args.num_nodes * args.seq_len
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.step =  args.seq_len

    # Load your dataset here
    # data = load_dataset(...)
    if args.mode == "in-sample":
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)
    else:
        raise  ValueError
    model = NBeatsModel(args, args.enc_input_dim, args.output_dim, num_blocks=3, hidden_dim=64)
    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    args.data_loader = data["train_loader"]
    args.val_dataloader = data["val_loader"]
    args.test_dataloader = data["test_loader"]
    args.scalers = data["scalers"]
    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True)
    num_samples = data["x_train"].shape[0]
    val_samples = data["x_val"].shape[0]
    test_samples = data["x_test"].shape[0]
    args.scalers = data["scalers"]
    args.train_iters = math.ceil(num_samples / args.batch_size)
    args.val_iters = math.ceil(val_samples / args.batch_size)
    args.test_iters = math.ceil(test_samples / args.batch_size)
    args.early_stopper = EarlyStopper(tolerance=10, min_delta=0.01)
    # training_iter_time = num_samples / batch_size
    # len_epoch = math.ceil(num_samples / batch_size)
    args.len_epoch = 150 #500
    trainer = Trainer(model, args, logger)
    total_train_time = trainer.train() # annotate when testing
    trainer.test(total_train_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='nbeats', help='file name')
    args = parser.parse_args()
    args.filename = "nbeat"
    args.batch_size = 64
    args.enc_input_dim = 3  # encoder network input size, can be 1 or 3
    args.dec_input_dim = 3  # decoder input
    args.features = 3      # actual features
    args.max_diffusion_step = 2
    args.num_nodes = 35
    args.num_rnn_layers = 2
    args.rnn_units = 64
    args.seq_len = 12
    args.output_dim = 3
    args.device = device
    args.filter_type = "dual_random_walk"
    args.max_grad_norm = 5
    args.cl_decay_steps = 2000
    args.cl = False
    args.loss_func = "none"
    main(args)