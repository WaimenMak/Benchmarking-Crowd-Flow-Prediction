# -*- coding: utf-8 -*-
# @Time    : 16/04/2023 11:54
# @Author  : mmai
# @FileName: MTGNN
# @Software: PyCharm

from layer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import build_graph, load_dataset, EarlyStopper
import math
import logging
import argparse
from lib.train_test import train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=3, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self._output_dim = 3
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


if __name__ == "__main__":
    training_iter_time = 0
    # total_train_time = 0
    G = build_graph()
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="coo")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='mtgnn', help='file name')
    # args = parser.parse_args()

    parser.add_argument('--device',type=str,default='cpu',help='')
    parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

    parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
    parser.add_argument('--num_nodes',type=int,default=35,help='number of nodes/variables')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--subgraph_size',type=int,default=20,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
    parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=128,help='end channels')


    parser.add_argument('--in_dim',type=int,default=3,help='inputs dimension')
    parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
    parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

    parser.add_argument('--layers',type=int,default=3,help='number of layers')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--clip',type=int,default=5,help='clip')
    parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
    parser.add_argument('--step_size2',type=int,default=100,help='step_size')


    parser.add_argument('--epochs',type=int,default=100,help='')
    parser.add_argument('--print_every',type=int,default=50,help='')
    parser.add_argument('--seed',type=int,default=101,help='random seed')
    parser.add_argument('--save',type=str,default='./save/',help='save path')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')

    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

    parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

    parser.add_argument('--runs',type=int,default=10,help='number of runs')

    args = parser.parse_args()

    file_handler = logging.FileHandler("./result/train "+args.filename+".log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


    if args.mode == "in-sample":
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")
    args.data_loader = data["train_loader"]
    args.val_dataloader = data["val_loader"]
    args.test_dataloader = data["test_loader"]
    args.scalers = data["scalers"]
    args.max_grad_norm = 5
    args.cl_decay_steps = 2000
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
              device, predefined_A=adj_mat,
              dropout=args.dropout, subgraph_size=args.subgraph_size,
              node_dim=args.node_dim,
              dilation_exponential=args.dilation_exponential,
              conv_channels=args.conv_channels, residual_channels=args.residual_channels,
              skip_channels=args.skip_channels, end_channels= args.end_channels,
              seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
              layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True)
    args.num_samples = data["x_train"].shape[0]
    args.val_samples = data["x_val"].shape[0]
    args.test_samples = data["x_test"].shape[0]
    args.train_iters = math.ceil(args.num_samples / args.batch_size)
    args.val_iters = math.ceil(args.val_samples / args.batch_size)
    args.test_iters = math.ceil(args.test_samples / args.batch_size)
    args.early_stopper = EarlyStopper(tolerance=15, min_delta=0.01)
    # training_iter_time = num_samples / batch_size
    # len_epoch = math.ceil(num_samples / batch_size)
    args.len_epoch = 100  #500
    total_train_time = train(model, args, logger)