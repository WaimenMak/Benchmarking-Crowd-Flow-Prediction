# -*- coding: utf-8 -*-
"""
Created on 04/01/2023 14:25

@Author: mmai
@FileName: GAT_traffic.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import build_graph, load_dataset, EarlyStopper
import math

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)  # [B, N, D]*[B, D, N]->[B, N, N]      x(i)^T * x(j)

        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)   # [B, N, N]
        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D]


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()

        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)

        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, graph, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)
        self.graph = graph
        self._output_dim = out_c
        # self.subnet = [GATSubNet(...) for _ in range(T)]

    def forward(self, data):
        # graph = data["graph"][0].to(device)  # [N, N]
        # flow = data["flow_x"]  # [B, N, T, C]
        # flow = flow.to(device)
        flow = data
        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)  # [B, N, T * C]

        prediction = self.subnet(flow, self.graph).unsqueeze(1)  # [B, 1, N, C]

        return prediction

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
    num_heads = 5
    hidden_dim = 64
    seq_len = 12
    output_dim = 3
    device = "cpu"

    graph = torch.zeros([num_nodes, num_nodes]).detach()
    graph[adj_mat.row, adj_mat.col] = 1
    max_grad_norm = 5
    cl_decay_steps = 2000
    data = load_dataset("./dataset", batch_size=64, test_batch_size=64)
    data_loader = data["train_loader"]
    val_dataloader = data["val_loader"]
    test_dataloader = data["test_loader"]
    model = GATNet(graph, enc_input_dim * seq_len, hidden_dim, output_dim, num_heads)

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
            output = model(data.permute(0, 2, 1, 3).reshape([batch_size, num_nodes, -1])) # [bc, node, output_dim]


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

            torch.save(model.state_dict(), "./result/gat_traffic.pt")
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
