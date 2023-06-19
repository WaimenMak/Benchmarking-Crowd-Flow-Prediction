# -*- coding: utf-8 -*-
"""
Created on 28/12/2022 11:33

@Author: mmai
@FileName: GAT.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import build_graph, load_dataset, EarlyStopper, init_seq2seq
import math
import random
import logging
import argparse
from lib.train_test import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GATGRULayer(nn.Module):
    def __init__(self, adj, seq_len, input_dim, output_dim, nodes, num_layer, dropout=0.5):
        super(GATGRULayer, self).__init__()
        self.adj = adj
        self.nodes = nodes
        # self.bc = bc
        self.feat = input_dim  #feat = 3
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layer = num_layer

        self.rnn = nn.GRU(input_dim, output_dim, num_layer, dropout=dropout) # could be change to RNN encoder

        self.atten_W = nn.Parameter(torch.FloatTensor(size=(2*self.output_dim, 1)))
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.apply(init_seq2seq)
        nn.init.xavier_normal_(self.atten_W.data, gain=1.414)
        # nn.init.normal_(self.W)
        # nn.init.normal_(self.atten_W)

    def edge_attention_concatenate(self, z):
        bc = z.shape[0]
        b = z.repeat([1, 1, self.nodes]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        c = z.repeat([1, self.nodes, 1]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        e = torch.cat([b,c],dim=3).reshape(self.bc, -1, 2*self.output_dim) # [bc, node*node, output_dim*2]
        mask = torch.zeros([self.nodes, self.nodes])
        mask[self.adj.row, self.adj.col] = 1
        # mask = mask.repeat([self.bc, 1, 1])
        # atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(self.bc, self.nodes, self.nodes)) * mask.unsqueeze(0) #[bc, node, node] batch attention scores
        atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(self.bc, self.nodes, self.nodes)) # not just consider neighbors
        atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat

    def edge_attention_innerprod(self, z):
        atten_mat = torch.bmm(z, z.transpose(2, 1)) #[bc node feat] * [bc feat node]
        mask = torch.zeros([self.nodes, self.nodes])
        mask[self.adj.row, self.adj.col] = 1
        atten_mat = self.leaky_relu(atten_mat) * mask.unsqueeze(0)
        # atten_mat = self.leaky_relu(atten_mat)
        atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat

    def forward(self, h):
        # h: [seq_len, bc, node, feature]
        bc = h.shape[1]
        h_agg = []
        ouput, z = self.rnn(h.reshape(self.seq_len, bc*self.nodes, self.feat)) # bc*nodes  # z = Wh  #[bc*node, output_dim]
        z = z.reshape(self.num_layer, bc, self.nodes, self.output_dim)  # [bc, nodes, output_dim]
        # atten_mat = self.edge_attention_concatenate(z)
        for z_ in z:
            atten_mat = self.edge_attention_innerprod(z_)
            # normallization
            # atten_mat = F.normalize(atten_mat, p=1, dim=2)
            atten_mat = F.softmax(atten_mat, dim=2)
            # h_agg = torch.matmul(atten_mat, z)  # [bc, node, output_dim]
            h_agg.append(torch.bmm(atten_mat, z_))
        return h_agg

# class GATGRULayer_dec(nn.Module):
#     def __init__(self, feature_size, num_hiddens, num_layer, output_dim, dropout=0.01):
#         super().__init__()
#         self.dense = nn.Linear(num_hiddens, output_dim)
#         self.rnn = nn.GRU(feature_size + num_hiddens, num_hiddens, num_layer, dropout=dropout)
#         self.apply(init_seq2seq)
#
#     def forward(self, X, enc_state, teacher_forcing_ratio=0.5):
#         # X shape: (seq, batch_size, num_steps), target
#         # embs shape: (num_steps, batch_size, embed_size)
#         # context shape: (batch_size, num_hiddens)
#         context = enc_state[-1]
#         last_hidden_state = enc_state
#         # Broadcast context to (num_steps, batch_size, num_hiddens)
#         context = context.repeat(X.shape[0], 1, 1)
#         # Concat at the feature dimension
#         embs_and_context = torch.cat((X, context), -1)  # seq, bc, features+hidden_dims
#         outputs_list = []
#         current_input = embs_and_context[0, ...].unsqueeze(dim=0)
#         for t in range(1, X.shape[0]): #seq_len: 13
#             outputs, state = self.rnn(current_input, last_hidden_state) #seq, bc, num_hiddens, hidden_state: 2 layers
#             last_hidden_state = state
#             outputs = self.dense(outputs)
#             outputs_list.append(outputs.swapaxes(0, 1).squeeze())
#
#             teacher_force = random.random() < teacher_forcing_ratio  # a bool value
#             current_input = (X[t, ...].unsqueeze(0) if teacher_force else outputs)
#             current_input = torch.cat((current_input, context[t, ...].unsqueeze(0)), -1)
#
#
#         # outputs shape: (batch_size, num_steps, vocab_size)
#         # state shape: (num_layers, batch_size, num_hiddens)
#         return outputs_list, state


class GATLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, nodes):
        super(GATLayer, self).__init__()
        self.adj = adj
        self.nodes = nodes
        # self.bc = bc
        self.feat = input_dim
        self.output_dim = output_dim

        self.W = nn.Parameter(torch.FloatTensor(size=(input_dim, output_dim))) # could be change to RNN encoder
        self.atten_W = nn.Parameter(torch.FloatTensor(size=(2*self.output_dim, 1)))
        self.leaky_relu = nn.LeakyReLU(0.1)
        # nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.atten_W, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.atten_W.data, gain=1.414)
        # nn.init.normal_(self.W)
        # nn.init.normal_(self.atten_W)

    def edge_attention_concatenate(self, z):
        bc = z.shape[0]
        b = z.repeat([1, 1, self.nodes]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        c = z.repeat([1, self.nodes, 1]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        e = torch.cat([b,c],dim=3).reshape(self.bc, -1, 2*self.output_dim) # [bc, node*node, output_dim*2]
        mask = torch.zeros([self.nodes, self.nodes])
        mask[self.adj.row, self.adj.col] = 1
        # mask = mask.repeat([self.bc, 1, 1])
        # atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(self.bc, self.nodes, self.nodes)) * mask.unsqueeze(0) #[bc, node, node] batch attention scores
        atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(bc, self.nodes, self.nodes)) # not just consider neighbors
        atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat

    def edge_attention_innerprod(self, z):
        atten_mat = torch.bmm(z, z.transpose(2, 1)) #[bc node feat] * [bc feat node]
        mask = torch.zeros([self.nodes, self.nodes])
        mask[self.adj.row, self.adj.col] = 1
        atten_mat = self.leaky_relu(atten_mat) * mask.unsqueeze(0) # one hop
        # atten_mat = self.leaky_relu(atten_mat)
        atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat

    def forward(self, h):
        '''
        output_dim is the out dimenstion after original vector multiplied by W
        :param graph:
        :param data: [bc, seq, node, feature]  bc: shape[0], node: shape[2]
        :return:
        '''
        # h = data.transpose(0, 2, 1, 3).view(self.bc, self.nodes, -1) #[bc, node, feat]
        bc = h.shape[0]
        z = torch.matmul(h.reshape(bc*self.nodes, self.feat), self.W)  # z = Wh  #[bc*node, output_dim]
        z = z.reshape(bc, self.nodes, self.output_dim)  # [bc, nodes, output_dim]
        # atten_mat = self.edge_attention_concatenate(z)
        atten_mat = self.edge_attention_innerprod(z)
        # normallization
        # atten_mat = F.normalize(atten_mat, p=1, dim=2)
        atten_mat = F.softmax(atten_mat, dim=2)
        # h_agg = torch.matmul(atten_mat, z)  # [bc, node, output_dim], matmul could broadcast
        h_agg = torch.bmm(atten_mat, z)
        return h_agg

class MultiHeadGATLayer(nn.Module):
    def __init__(self, adj, seq_len, input_dim, output_dim, nodes, num_heads, rnn_num_layer, type='rnn', merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.merge = merge
        for i in range(num_heads):
            if type == 'rnn':
                self.heads.append(GATGRULayer(adj, seq_len, input_dim, output_dim, nodes, rnn_num_layer))
            elif type == "linear":
                self.heads.append(GATLayer(adj, input_dim, output_dim, nodes))

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            if isinstance(head_outs[0], list):
                result = [torch.cat(head, dim=2) for head in zip(*head_outs)]  # [len = rnn layers] concate each hidden state of each rnn layer
                return result   # two context for RNN after attention
            else:
                return torch.cat(head_outs, dim=2)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))

class GATEncoder(nn.Module):
    def __init__(self, g, seq_len, feature_size, hidden_dim, out_dim, nodes, num_heads, num_layer):
        super(GATEncoder, self).__init__()
        self._output_dim = out_dim
        self.layer1 = MultiHeadGATLayer(g, seq_len, feature_size, hidden_dim, nodes, num_heads, rnn_num_layer=num_layer)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, seq_len, hidden_dim * num_heads,
                                        output_dim=hidden_dim, nodes=nodes, num_heads=1, rnn_num_layer=num_layer, type="linear")  # hidden_dim = size of state

    def forward(self, h):
        '''

        :param h:  [bc, node, feat]
        :return:  [bc, node, output_dim]
        '''
        h_list = []
        h = self.layer1(h)
        for hagg in h:
            hagg = F.elu(hagg)
            h_tilde = self.layer2(hagg)
            h_tilde = h_tilde.unsqueeze(1)
            h_list.append(h_tilde)

        return torch.cat(h_list, dim=1) #return list of hidden state [bc, node, 1, output_dim]

class GATDecoder(nn.Module):
    def __init__(self, feature_size, num_hiddens, num_layer, output_dim, dropout=0.5):
        super().__init__()
        self.dense = nn.Linear(num_hiddens, output_dim)
        self.rnn = nn.GRU(feature_size + num_hiddens, num_hiddens, num_layer, dropout=dropout)  #inpusize: feature + numheads
        self.num_hiddens = num_hiddens
        self.apply(init_seq2seq)

    def forward(self, X, enc_state, teacher_forcing_ratio=0.5):
        # X shape: (seq, batch_size, num_steps), target
        # embs shape: (num_steps, batch_size, embed_size)
        # enc_state: (batch_size, 1, nodes, num_hiddens)
        enc_state = enc_state.transpose(1, 0).reshape(-1, X.shape[1] * X.shape[2], self.num_hiddens)
        context = enc_state[-1]
        last_hidden_state = enc_state
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(X.shape[0], 1, 1) #[seq_len, bc, node, hidden of rnn]
        # Concat at the feature dimension
        X = X.reshape(-1, X.shape[1] * X.shape[2], X.shape[3]) #[seq_len, bc * nodes, features]
        embs_and_context = torch.cat((X, context), -1)  # seq, bc, features+hidden_dims
        outputs_list = []
        current_input = embs_and_context[0, ...].unsqueeze(dim=0)
        for t in range(X.shape[0]): #seq_len: 13
            outputs, state = self.rnn(current_input, last_hidden_state) #seq, bc*nodes, num_hiddens; last_hidden_state: 2 layers
            last_hidden_state = state
            outputs = self.dense(outputs)
            outputs_list.append(outputs.swapaxes(0, 1).squeeze())

            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (X[t, ...].unsqueeze(0) if teacher_force else outputs)
            current_input = torch.cat((current_input, context[t, ...].unsqueeze(0)), -1)


        # outputs shape: (batch_size, num_steps, vocab_size)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs_list[1:], state


class GATSeq2seq(nn.Module):
    def __init__(self, g, args):
        super(GATSeq2seq, self).__init__()
        self._num_rnn_layers = args.num_rnn_layers  # should be 2
        self._rnn_units = args.rnn_units  # should be 64
        self._seq_len = args.seq_len  # should be 12
        # use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # should be true
        self._output_dim = args.output_dim  # should be 3: in out flow. overall.
        self.enc_input_dim = args.enc_input_dim
        self.GO_Symbol = torch.zeros(1, args.batch_size, args.num_nodes, args.enc_input_dim).to(args.device)
        # self.batch_size = batch_size

        # self.GO_Symbol = torch.zeros(1, batch_size, enc_input_dim).to(device)  #1 for seq length
        self.encoder = GATEncoder(g=g, seq_len= self._seq_len, feature_size=args.enc_input_dim, hidden_dim=self._rnn_units,
                                  out_dim=args.output_dim, nodes=g.shape[0], num_heads=args.num_heads,
                                  num_layer=self._num_rnn_layers)

        self.decoder = GATDecoder(feature_size=args.dec_input_dim, num_hiddens=self._rnn_units,
                                  num_layer=self._num_rnn_layers, output_dim=self._output_dim, dropout=0.5)
    def forward(self, source, target, teacher_forcing_ratio):
         # the size of source/target would be (64, 12, 207, 2)self.GO_Symbol = torch.zeros(1, batch_size, num_nodes, 3).to(device)
        # GO_Symbol = torch.zeros(1, target.size(0), self.enc_input_dim).to(device)
        source = torch.transpose(source, dim0=0, dim1=1)
        target = torch.transpose(target[..., :self._output_dim], dim0=0, dim1=1)
        target = torch.cat([self.GO_Symbol, target], dim=0)

        # initialize the hidden state of the encoder
        # init_hidden_state = self.encoder.init_hidden(self._batch_size).to(device)

        # last hidden state of the encoder is the context
        # _, context = self.encoder(source, init_hidden_state)  # (num_layers, batch, outdim/num_hiddens)
        context = self.encoder(source)

        outputs, _ = self.decoder(target, context, teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = torch.stack(outputs)
        # the elements of the first time step of the outputs are all zeros.
        outputs = outputs.swapaxes(0, 1)
        outputs = outputs.reshape([args.batch_size, args.num_nodes, args.seq_len, -1]).permute(0, 2, 1, 3)

        return  outputs # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 3)

def main(args):
    G = build_graph()
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="csr")
    adj_mat.setdiag(1)
    args.batch_size = 64
    args.enc_input_dim = 3  # encoder network input size, can be 1 or 3
    args.dec_input_dim = 3  # decoder input
    args.num_nodes = 35
    args.num_rnn_layers = 2
    args.num_heads = 3
    args.rnn_units = 64
    args.seq_len = 12
    args.output_dim = 3
    args.max_grad_norm = 5
    args.cl_decay_steps = 2000

    #File handeler
    file_handler = logging.FileHandler("./result/train "+args.filename+".log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    #Load data
    if args.mode == "in-sample":
        data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    elif args.mode == "ood":
        data = load_dataset("./ood_dataset", batch_size=64, test_batch_size=64)

    #Begin training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")
    args.data_loader = data["train_loader"]
    args.val_dataloader = data["val_loader"]
    args.test_dataloader = data["test_loader"]
    args.scalers = data["scalers"]
    model = GATSeq2seq(adj_mat, args)
    args.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True)
    args.num_samples = data["x_train"].shape[0]
    args.val_samples = data["x_val"].shape[0]
    args.test_samples = data["x_test"].shape[0]
    args.train_iters = math.ceil(args.num_samples / args.batch_size)
    args.val_iters = math.ceil(args.val_samples / args.batch_size)
    args.test_iters = math.ceil(args.test_samples / args.batch_size)
    args.early_stopper = EarlyStopper(tolerance=15, min_delta=0.01)

    args.len_epoch = 150  #500
    trainer = Trainer(model, args, logger)
    total_train_time = trainer.train()
    trainer.test(total_train_time)

if __name__ == "__main__":
    training_iter_time = 0
    # total_train_time = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='gatrnn', help='file name')
    args = parser.parse_args()
    G = build_graph()
    # adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="coo")   # v1
    adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="csr") # v2
    adj_mat.setdiag(1)                                             # v2
    args.batch_size = 64
    args.enc_input_dim = 3  # encoder network input size, can be 1 or 3
    args.dec_input_dim = 3  # decoder input
    # max_diffusion_step = 2
    args.num_nodes = 35
    args.num_rnn_layers = 2
    args.num_heads = 3
    args.rnn_units = 64
    args.seq_len = 12
    args.output_dim = 3
    # args.device = "cpu"

    args.max_grad_norm = 5
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
    model = GATSeq2seq(adj_mat, args)
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
    # best_val = float('inf')
    # train_list, val_list, test_list = [], [], []
    # model.to(device)
    # for epoch in range(len_epoch):
    #     model.train()
    #     train_rmse_loss = 0
    #     # training_time += train_epoch_time
    #     '''
    #     code from dcrnn_trainer.py, _train_epoch()
    #     '''
    #     start_time = time.time()  # record the start time of each epoch
    #     total_loss = 0
    #     # total_metrics = np.zeros(len(metrics))
    #     for batch_idx, (data, target) in enumerate(data_loader.get_iterator()):
    #         data = torch.FloatTensor(data)
    #         # data = data[..., :1]
    #         target = torch.FloatTensor(target)
    #         label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #         data, target = data.to(device), target.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         #data [bc, seq, node, feature]
    #         global_step = (epoch+1 - 1) * len_epoch + batch_idx
    #         teacher_forcing_ratio = _compute_sampling_threshold(global_step, cl_decay_steps)
    #
    #         #data [bc, seq, node, feature]
    #         # for i in range(num_nodes):
    #         # output = model(data.permute(0, 2, 1, 3).reshape([-1, seq_len, output_dim]), target.permute(0, 2, 1, 3).reshape(
    #         #     [-1, seq_len, output_dim]), teacher_forcing_ratio)
    #         # data [bc, seq_len, feature 105]
    #         output = model(data, target, teacher_forcing_ratio)
    #         output = output.reshape([batch_size, num_nodes, seq_len, -1]).permute(0, 2, 1, 3)
    #
    #         # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
    #         # loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
    #         loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
    #         train_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)] # training metric for each step
    #
    #         loss = sum(loss_sup_seq) / len(loss_sup_seq) / batch_size
    #         train_rmse = sum(train_rmse) / len(train_rmse) / batch_size
    #         # loss = loss(output.cpu(), label)  # loss is self-defined, need cpu input
    #         loss.backward()
    #         # add max grad clipping
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    #         optimizer.step()
    #         training_iter_time = time.time() - start_time
    #         total_train_time += training_iter_time
    #
    #         # writer.set_step((epoch - 1) * len_epoch + batch_idx)
    #         # writer.add_scalar('loss', loss.item())
    #         # total_loss += loss.item()
    #         # train_mse_loss += loss.item()
    #         train_rmse_loss += train_rmse.item()  #metric  sum of each node and each iteration
    #         # total_metrics += _eval_metrics(output.detach().numpy(), label.numpy()
    #
    #         # if batch_idx % log_step == 0:
    #         #     logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
    #         #         epoch,
    #         #         _progress(batch_idx),
    #         #         loss.item())
    #
    #         if batch_idx == len_epoch:
    #             break
    #
    #     train_rmse_loss = train_rmse_loss / train_iters
    #     # print(f"Epoch: {epoch}, train_RMSE: {train_rmse_loss}")
    #     logger.info(f"Epoch [{epoch+1}/{len_epoch}], train_RMSE: {train_rmse_loss:.4f}")
    #
    #
    #
    #     # validation
    #     if epoch % 5 == 0 and epoch != 0 :
    #         val_mse_loss = 0
    #         val_mask_rmse_loss = []
    #         val_mask_mae_loss = []
    #         val_mask_mape_loss = []
    #         # validation
    #         model.eval()
    #         with torch.no_grad():
    #             for batch_idx, (data, target) in enumerate(val_dataloader.get_iterator()):
    #                 data = torch.FloatTensor(data)
    #                 # data = data[..., :1]
    #                 target = torch.FloatTensor(target)
    #                 label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #                 data, target = data.to(device), target.to(device)
    #
    #                 # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0) # 0:teacher forcing rate
    #                 output = model(data, target, teacher_forcing_ratio=0)
    #                 output = output.reshape([batch_size, num_nodes, seq_len, -1]).permute(0, 2, 1, 3)
    #                 # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
    #                 # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
    #                 #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)
    #
    #                 val_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]   # 12 graphs
    #                 val_rmse = sum(val_rmse) / len(val_rmse) / batch_size
    #
    #                 val_mse_loss += val_rmse.item()
    #                 val_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
    #                 val_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
    #                 val_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))
    #
    #
    #         val_mse_loss = val_mse_loss / val_iters
    #         val_mask_rmse_loss = np.mean(val_mask_rmse_loss)
    #         val_mask_mape_loss = np.mean(val_mask_mape_loss)
    #         val_mask_mae_loss = np.mean(val_mask_mae_loss)
    #         # print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
    #         logger.info(f"Epoch [{epoch+1}/{len_epoch}], val_RMSE: {val_mse_loss:.4f}, val_MASK_RMSE: {val_mask_rmse_loss:.4f}, val_MASK_MAPE: {val_mask_mae_loss:.4f}, val_MASK_MAE: {val_mask_mape_loss:.4f}")
    #         train_list.append(train_rmse_loss)
    #         val_list.append(val_mse_loss)
    #         if args.mode == "in-sample":
    #             np.save("./result/"+args.filename+"_train_loss.npy", train_list)
    #             np.save("./result/"+args.filename+"_val_loss.npy", val_list)
    #             if val_mse_loss < best_val:
    #                 best_val = val_mse_loss
    #                 torch.save(model.state_dict(), "./result/"+args.filename+".pt")
    #         else:
    #             np.save("./result/"+args.filename+"_train_loss_ood.npy", train_list)
    #             np.save("./result/"+args.filename+"_val_loss_ood.npy", val_list)
    #             if val_mse_loss < best_val:
    #                 best_val = val_mse_loss
    #                 torch.save(model.state_dict(), "./result/"+args.filename+"_ood.pt")
    #         if early_stopper.early_stop(val_mse_loss):
    #             # break
    #             pass
    # # Testing
    # test_mse_loss = 0
    # test_mask_rmse_loss = []
    # test_mask_mae_loss = []
    # test_mask_mape_loss = []
    # if args.mode == "in-sample":
    #     model.load_state_dict(torch.load("./result/"+args.filename+".pt"))
    # elif args.mode == "ood":
    #     model.load_state_dict(torch.load("./result/"+args.filename+"_ood.pt"))
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(test_dataloader.get_iterator()):
    #         data = torch.FloatTensor(data)
    #         # data = data[..., :1]
    #         target = torch.FloatTensor(target)
    #         label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #         data, target = data.to(device), target.to(device)
    #
    #         # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0)
    #         # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
    #         #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)
    #         output = model(data, target, teacher_forcing_ratio=0)
    #         output = output.reshape([batch_size, num_nodes, seq_len, -1]).permute(0, 2, 1, 3)
    #         # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
    #         test_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]
    #         test_rmse = sum(test_rmse) / len(test_rmse) / batch_size
    #
    #         test_mse_loss += test_rmse.item()
    #         test_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
    #         test_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
    #         test_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))
    #
    # test_mse_loss = test_mse_loss / test_iters
    # test_mask_rmse_loss = np.mean(test_mask_rmse_loss)
    # test_mask_mape_loss = np.mean(test_mask_mape_loss)
    # test_mask_mae_loss = np.mean(test_mask_mae_loss)
    # print(f"test_MSE: {test_mse_loss}, Time: {total_train_time}")
    test_mse_loss, test_mask_rmse_loss, test_mask_mae_loss, test_mask_mape_loss = test(model, args, logger)
    logger.info(f"testing method: {args.mode}, test_RMSE: {test_mse_loss:.4f}, test_MASK_RMSE: {test_mask_rmse_loss:.4f}, test_MASK_MAE: {test_mask_mae_loss:.4f}, test_MASK_MAPE: {test_mask_mape_loss:.4f}, Time: {total_train_time:.4f}")

