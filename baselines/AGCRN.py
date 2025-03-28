# -*- coding: utf-8 -*-
# @Time    : 15/04/2023 12:30
# @Author  : mmai
# @FileName: AGCRN
# @Software: PyCharm

# from model.AGCRNCell import AGCRNCell
import torch
import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
from lib.utils import load_dataset, EarlyStopper, build_graph
import math
import argparse
import logging
from lib.train_test import Trainer
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)



class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.enc_input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self._output_dim = args.output_dim
        self.horizon = args.seq_len
        self.num_layers = args.num_rnn_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.enc_input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_rnn_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.seq_len * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output


def main(args):
    G = build_graph()
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="csr")
    adj_mat.setdiag(1)
    args.default_graph = adj_mat
    args.batch_size = 32
    args.enc_input_dim = 3  # encoder network input size, can be 1 or 3
    args.dec_input_dim = 3  # decoder input
    # args.max_diffusion_step = 2
    args.num_nodes = 35
    args.num_rnn_layers = 2
    args.rnn_units = 64
    args.seq_len = 12
    args.output_dim = 3
    args.features = 3      # actual features
    args.device = device
    args.embed_dim = 10
    args.cheb_k = 3
    args.max_grad_norm = 5
    args.lr = 0.005
    args.cl_decay_steps = 2000

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
    model = AGCRN(args)
    # model.train()
    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1.0e-3, amsgrad=True)
    args.num_samples = data["x_train"].shape[0]
    args.val_samples = data["x_val"].shape[0]
    args.test_samples = data["x_test"].shape[0]
    args.train_iters = math.ceil(args.num_samples / args.batch_size)
    args.val_iters = math.ceil(args.val_samples / args.batch_size)
    args.test_iters = math.ceil(args.test_samples / args.batch_size)
    args.early_stopper = EarlyStopper(tolerance=10, min_delta=0.01)
    # training_iter_time = num_samples / batch_size
    # len_epoch = math.ceil(num_samples / batch_size)

    trainer = Trainer(model, args, logger)
    args.len_epoch = 100  #500
    total_train_time = trainer.train()

   # total_train_time = trainer.train()  # annotate for testing
    trainer.test(total_train_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='agcrn', help='file name')
    args = parser.parse_args()
    args.filename = "agcrn"
    training_iter_time = 0
    total_train_time = 0
    G = build_graph()
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="csr")
    adj_mat.setdiag(1)
    args.default_graph = adj_mat
    args.batch_size = 64
    args.enc_input_dim = 3  # encoder network input size, can be 1 or 3
    args.dec_input_dim = 3  # decoder input
    # args.max_diffusion_step = 2
    args.num_nodes = 35
    args.num_rnn_layers = 2
    args.rnn_units = 32
    args.seq_len = 12
    args.output_dim = 3
    args.device = device
    args.embed_dim = 10
    args.cheb_k = 2
    args.max_grad_norm = 5
    args.lr = 0.0005
    args.cl_decay_steps = 2000
    args.cl = False
    args.loss_func = "none"
    args.step = 12



    file_handler = logging.FileHandler("../result/train "+args.filename+".log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


    if args.mode == "in-sample":
        data = load_dataset("../dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("../ood_dataset", batch_size=64, test_batch_size=64)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")
    args.data_loader = data["train_loader"]
    args.val_dataloader = data["val_loader"]
    args.test_dataloader = data["test_loader"]
    args.scalers = data["scalers"]
    model = AGCRN(args)
    # model.train()
    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1.0e-3, amsgrad=True)
    args.num_samples = data["x_train"].shape[0]
    args.val_samples = data["x_val"].shape[0]
    args.test_samples = data["x_test"].shape[0]
    args.train_iters = math.ceil(args.num_samples / args.batch_size)
    args.val_iters = math.ceil(args.val_samples / args.batch_size)
    args.test_iters = math.ceil(args.test_samples / args.batch_size)
    args.early_stopper = EarlyStopper(tolerance=10, min_delta=0.01)
    # training_iter_time = num_samples / batch_size
    # len_epoch = math.ceil(num_samples / batch_size)

    trainer = Trainer(model, args, logger)
    args.len_epoch = 100  #500
    total_train_time = trainer.train()

   # total_train_time = trainer.train()  # annotate for testing
    trainer.test(total_train_time)