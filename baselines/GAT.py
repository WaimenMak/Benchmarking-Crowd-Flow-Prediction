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
from lib.utils import build_graph, load_dataset, EarlyStopper
import math

class GATLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, nodes, bc):
        super(GATLayer, self).__init__()
        self.adj = adj
        self.nodes = nodes
        self.bc = bc
        self.feat = input_dim  # seq_len * features
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
        b = z.repeat([1, 1, self.nodes]).reshape([self.bc, self.nodes, self.nodes, self.output_dim]) #
        c = z.repeat([1, self.nodes, 1]).reshape([self.bc, self.nodes, self.nodes, self.output_dim]) #
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
        '''
        h: [bc, node, seq_len*feature]
        output_dim is the out dimenstion after original vector multiplied by W
        :param graph:
        :param data: [bc, seq, node, feature]  bc: shape[0], node: shape[2]
        :return:
        '''
        # h = data.transpose(0, 2, 1, 3).view(self.bc, self.nodes, -1) #[bc, node, feat]
        z = torch.matmul(h.reshape(self.bc*self.nodes, self.feat), self.W)  # z = Wh  #[bc*node, output_dim]
        z = z.reshape(self.bc, self.nodes, self.output_dim)  # [bc, nodes, output_dim]
        # atten_mat = self.edge_attention_concatenate(z)
        atten_mat = self.edge_attention_innerprod(z)
        # normallization
        # atten_mat = F.normalize(atten_mat, p=1, dim=2)
        atten_mat = F.softmax(atten_mat, dim=2)
        h_agg = torch.matmul(atten_mat, z)  # [bc, node, output_dim]

        return h_agg

class MultiHeadGATLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, nodes, bc, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.merge = merge
        for i in range(num_heads):
            self.heads.append(GATLayer(adj, input_dim, output_dim, nodes, bc))

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=2)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, seq_len, feature_size, hidden_dim, out_dim, nodes, bc, num_heads):
        super(GAT, self).__init__()
        self._output_dim = out_dim
        self.layer1 = MultiHeadGATLayer(g, seq_len*feature_size, hidden_dim, nodes, bc, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, nodes, bc, 1)

    def forward(self, h):
        '''

        :param h:  [bc, node, feat]
        :return:  [bc, node, output_dim]
        '''
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h.unsqueeze(1) #return [bc, node, 1, output_dim]

if __name__ == "__main__":
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
    num_heads = 8
    hidden_dim = 64
    seq_len = 12
    output_dim = 3
    device = "cpu"

    max_grad_norm = 5
    cl_decay_steps = 2000
    data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    data_loader = data["train_loader"]
    val_dataloader = data["val_loader"]
    test_dataloader = data["test_loader"]
    model = GAT(adj_mat, seq_len, enc_input_dim, hidden_dim, output_dim, num_nodes, batch_size, num_heads)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True)
    num_samples = data["x_train"].shape[0]
    val_samples = data["x_val"].shape[0]
    test_samples = data["x_test"].shape[0]
    train_iters = math.ceil(num_samples / batch_size)
    val_iters = math.ceil(val_samples / batch_size)
    test_iters = math.ceil(test_samples / batch_size)
    early_stopper = EarlyStopper(tolerance=2, min_delta=0.01)
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
            output = model(data.permute(0, 2, 1, 3).reshape([batch_size, num_nodes, -1])) # [bc, 1, node, output_dim]


            # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
            # loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
            loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(1)]  #training loss function
            train_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)] # training metric for each step

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

                    val_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]   # 12 graphs
                    val_rmse = sum(val_rmse) / len(val_rmse) / batch_size

                    val_mse_loss += val_rmse.item()



            val_mse_loss = val_mse_loss / val_iters

            print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
            train_list.append(train_rmse_loss)
            val_list.append(val_mse_loss)
            # np.save("./result/gat_train_loss.npy", train_list)
            # np.save("./result/gat_val_loss.npy", val_list)

            # torch.save(model.state_dict(), "./result/gat.pt")
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
            test_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(1)]
            test_rmse = sum(test_rmse) / len(test_rmse) / batch_size

            test_mse_loss += test_rmse.item()

    test_mse_loss = test_mse_loss / test_iters
    print(f"test_MSE: {test_mse_loss}, Time: {total_train_time}")
