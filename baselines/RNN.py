# -*- coding: utf-8 -*-
"""
Created on 04/12/2022 16:44

@Author: mmai
@FileName: RNN.py
@Software: PyCharm
"""
import torch.nn as nn
from lib.train_test import Trainer
import torch
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np
import random
from main import str_to_bool
from lib.utils import load_dataset, EarlyStopper,  init_seq2seq, _compute_sampling_threshold
import math
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
device = "cpu"
logger.info(f"Using device: {device}")



class RNNencoder(nn.Module):
    def __init__(self,feature_size, num_hiddens, num_layer, dropout=0.5):
        super().__init__()
        self.rnn = nn.GRU(feature_size, num_hiddens, num_layer, dropout=dropout)
        self.apply(init_seq2seq)
    def forward(self, X):
        # X: bc * num_steps
        # X = X.to(torch.float32)
        ouput, hidden_state = self.rnn(X)
        return ouput, hidden_state


class RNNdecoder(nn.Module):
    def __init__(self, feature_size, num_hiddens, num_layer, output_dim, dropout=0.5):
        '''

        :param feature_size:
        :param num_hiddens:  rnn_units
        :param num_layer:
        :param output_dim:
        :param dropout:
        '''
        super().__init__()
        self.dense = nn.Linear(num_hiddens, output_dim)
        self.rnn = nn.GRU(feature_size + num_hiddens, num_hiddens, num_layer, dropout=dropout)
        self.apply(init_seq2seq)

    def forward(self, X, enc_state, teacher_forcing_ratio=0.5):
        # X shape: (seq, batch_size, num_steps), target
        # embs shape: (num_steps, batch_size, embed_size)
        # context shape: (batch_size, num_hiddens)
        context = enc_state[-1]
        last_hidden_state = enc_state
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(X.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = torch.cat((X, context), -1)  # seq, bc, features+hidden_dims
        outputs_list = []
        current_input = embs_and_context[0, ...].unsqueeze(dim=0)
        # for t in range(1, X.shape[0]): #seq_len: 13
        for t in range(X.shape[0]): #seq_len: 13
            outputs, state = self.rnn(current_input, last_hidden_state) #seq, bc, num_hiddens, hidden_state: 2 layers
            last_hidden_state = state
            outputs = self.dense(outputs)
            outputs_list.append(outputs.swapaxes(0, 1).squeeze())

            teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            current_input = (X[t, ...].unsqueeze(0) if teacher_force else outputs)
            current_input = torch.cat((current_input, context[t, ...].unsqueeze(0)), -1)


        # outputs shape: (batch_size, num_steps, vocab_size)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs_list[1:], state


class Seq2seq(nn.Module):

    def __init__(self, args):
        super(Seq2seq, self).__init__()
        self._num_rnn_layers = args.num_rnn_layers  # should be 2
        self._rnn_units = args.rnn_units  # should be 64
        self._seq_len = args.seq_len  # should be 12
        # use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # should be true
        self._output_dim = args.output_dim  # should be 3: in out flow. overall.
        self.enc_input_dim = args.enc_input_dim
        self.dec_input_dim = args.dec_input_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.num_nodes = args.num_nodes

        # self.batch_size = batch_size

        # self.GO_Symbol = torch.zeros(1, batch_size, enc_input_dim).to(device)  #1 for seq length
        self.encoder = RNNencoder(feature_size=self.enc_input_dim, num_hiddens=self._rnn_units,
                                  num_layer=self._num_rnn_layers, dropout=0.5)

        self.decoder = RNNdecoder(feature_size=self.dec_input_dim, num_hiddens=self._rnn_units,
                                  num_layer=self._num_rnn_layers, output_dim=self._output_dim, dropout=0.5)

        # assert self.encoder.hid_dim == self.decoder.hid_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, source, target, teacher_forcing_ratio):
        # the size of source/target would be (64, 12, 35 * 3)
        source = torch.flatten(torch.FloatTensor(source), start_dim=2)
        target = torch.flatten(torch.FloatTensor(target), start_dim=2)
        GO_Symbol = torch.zeros(1, target.size(0), self.enc_input_dim).to(device)
        source = torch.transpose(source, dim0=0, dim1=1)
        target = torch.transpose(target[..., :self._output_dim], dim0=0, dim1=1)
        target = torch.cat([GO_Symbol, target], dim=0)

        # initialize the hidden state of the encoder
        # init_hidden_state = self.encoder.init_hidden(self._batch_size).to(device)

        # last hidden state of the encoder is the context
        # _, context = self.encoder(source, init_hidden_state)  # (num_layers, batch, outdim/num_hiddens)
        _, context = self.encoder(source)

        outputs, _ = self.decoder(target, context, teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = torch.stack(outputs)
        # the elements of the first time step of the outputs are all zeros.
        return outputs.swapaxes(0, 1).reshape([self.batch_size, self.seq_len, self.num_nodes, -1])  # (seq_length, batch_size, num_nodes*output_dim)  (64 12 35 3)



def main(args):
    # import time
    total_train_time = 0
    # parser = argparse.ArgumentParser()
    # for time calculating
    # training_iter_time = 0
    # total_train_time = 0
    # G = build_graph()
    # adj_mat = G.adjacency_matrix(transpose=False, scipy_fmt="csr")
    args.batch_size = 64
    args.num_nodes = 35
    # dec_input_dim = 3  # decoder input
    args.dec_input_dim = 3 * args.num_nodes  # decoder input
    args.enc_input_dim = 3 * args.num_nodes # encoder network input size, can be 1 or 3
    args.features = 3
    args.num_rnn_layers = 2
    args.rnn_units = 64
    args.seq_len = 12
    args.output_dim = 3 * args.num_nodes
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    else:
        raise  ValueError

    args.data_loader = data["train_loader"]
    args.val_dataloader = data["val_loader"]
    args.test_dataloader = data["test_loader"]
    args.scalers = data["scalers"]
    model = Seq2seq(args)
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
    args.len_epoch = 200 #500
    trainer = Trainer(model, args, logger)
    # total_train_time = trainer.train() # annotate when testing
    trainer.test(total_train_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='in-sample', help='dataset choice')
    parser.add_argument('--filename', type=str, default='rnn', help='file name')
    parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')
    parser.add_argument('--loss_func', type=str, default='none', help='loss func')
    args = parser.parse_args()

    main(args)
    # train_list, val_list, test_list = [], [], []
    # best_val = float('inf')
    # log_dir = "./result/"+args.filename+"_logs/"
    # writer = SummaryWriter(log_dir=log_dir)
    # for epoch in range(1, len_epoch):
    #     model.train()
    #     train_rmse_loss = 0
    #     # training_time += train_epoch_time
    #     '''
    #     code from dcrnn_trainer.py, _train_epoch()
    #     '''
    #     start_time = time.time()  # record the start time of each epoch
    #     total_loss = 0
    #     # total_metrics = np.zeros(len(metrics))
    #     for batch_idx, (data, org_target) in enumerate(data_loader.get_iterator()):
    #         data = torch.flatten(torch.FloatTensor(data), start_dim=2)
    #         # data = data[..., :1]
    #         target = torch.flatten(torch.FloatTensor(org_target), start_dim=2)
    #         # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #         label = torch.FloatTensor(org_target)
    #         data, target = data.to(device), target.to(device)
    #
    #         optimizer.zero_grad()
    # #
    #         # compute sampling ratio, which gradually decay to 0 during training
    #         global_step = (epoch - 1) * len_epoch + batch_idx
    #         teacher_forcing_ratio = _compute_sampling_threshold(global_step, cl_decay_steps)

            #data [bc, seq, node, feature]
            # for i in range(num_nodes):
            # output = model(data.permute(0, 2, 1, 3).reshape([-1, seq_len, output_dim]), target.permute(0, 2, 1, 3).reshape(
            #     [-1, seq_len, output_dim]), teacher_forcing_ratio)
            # data [bc, seq_len, feature 105]
            # output = model(data, target, teacher_forcing_ratio)
            # output = torch.transpose(output, 0, 1)  # back to (64, 12, 3)

            # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
            # output = output.reshape([batch_size, seq_len, num_nodes, label.shape[-1]]) # output_dim=3
            # loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
            # train_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(12)] # training metric for each step

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
    #             for batch_idx, (data, org_target) in enumerate(val_dataloader.get_iterator()):
    #                 data = torch.flatten(torch.FloatTensor(data), start_dim=2)
    #                 # data = data[..., :1]
    #                 target = torch.flatten(torch.FloatTensor(org_target), start_dim=2)
    #                 # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #                 label = torch.FloatTensor(org_target)
    #                 data, target = data.to(device), target.to(device)
    #
    #                 # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0) # 0:teacher forcing rate
    #                 # output = model(data.permute(0, 2, 1, 3).reshape([-1, seq_len, output_dim]), target.permute(0, 2, 1, 3).reshape(
    #                 #                 [-1, seq_len, output_dim]), teacher_forcing_ratio=0)
    #                 # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
    #                 output = model(data, target, teacher_forcing_ratio)
    #                 output = output.reshape([batch_size, seq_len, num_nodes, label.shape[-1]]) # output_dim=3
    #
    #                 val_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(12)]   # 12 graphs
    #                 val_rmse = sum(val_rmse) / len(val_rmse) / batch_size
    #
    #                 val_mse_loss += val_rmse.item()
    #                 for i in range(3): #3 feature num
    #                     output[..., i] = args.scalers[i].inverse_transform(output[..., i])
    #                     label[..., i] = args.scalers[i].inverse_transform(label[..., i])
    #                 val_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
    #                 val_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
    #                 val_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))
    #
    #
    #         val_mse_loss = val_mse_loss / val_iters
    #         val_mask_rmse_loss = np.mean(val_mask_rmse_loss)
    #         val_mask_mape_loss = np.mean(val_mask_mape_loss)
    #         val_mask_mae_loss = np.mean(val_mask_mae_loss)
    #
    #         # print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
    #         # logger.info(f"Epoch [{epoch+1}/{len_epoch}], val_RMSE: {val_mse_loss:.4f}")
    #         logger.info(f"Epoch [{epoch+1}/{len_epoch}], val_RMSE: {val_mse_loss:.4f}, val_MASK_RMSE: {val_mask_rmse_loss:.4f}, val_MASK_MAE: {val_mask_mae_loss:.4f}, val_MASK_MAPE: {val_mask_mape_loss:.4f}")
    #         train_RMSE: {train_rmse_loss}
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
    #         # # np.save("./result/rnn_train_loss.npy", train_list)
    #         # # np.save("./result/rnn_val_loss.npy", val_list)
    #         # np.save("./result/rnn_train_loss_ood.npy", train_list)
    #         # np.save("./result/rnn_val_loss_ood.npy", val_list)
    #
    #         # torch.save(model.state_dict(), "./result/rnn.pt")
    #         # torch.save(model.state_dict(), "./result/rnn_ood.pt")
    #         if early_stopper.early_stop(val_mse_loss):
    #             # break
    #             pass
    # # Testing
    # test_mse_loss = 0
    # test_mask_rmse_loss = []
    # test_mask_mae_loss = []
    # test_mask_mape_loss = []
    # half_test_mask_rmse_loss = []
    # half_test_mask_mae_loss = []
    # half_test_mask_mape_loss = []
    # end_test_mask_rmse_loss = []
    # end_test_mask_mae_loss = []
    # end_test_mask_mape_loss = []
    # if args.mode == "in-sample":
    #     model.load_state_dict(torch.load("./result/"+args.filename+".pt"))
    # elif args.mode == "ood":
    #     model.load_state_dict(torch.load("./result/"+args.filename+"_ood.pt"))
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, (data, org_target) in enumerate(test_dataloader.get_iterator()):
    #         data = torch.flatten(torch.FloatTensor(data), start_dim=2)
    #         # data = data[..., :1]
    #         target = torch.flatten(torch.FloatTensor(org_target), start_dim=2)
    #         # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
    #         label = torch.FloatTensor(org_target)
    #         data, target = data.to(device), target.to(device)
    #
    #         # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0)
    #         # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
    #         #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)
    #         output = model(data, target, teacher_forcing_ratio=0) # 0:teacher forcing rate
    #         output = output.reshape([batch_size, seq_len, num_nodes, 3])
    #         test_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(12)]
    #         test_rmse = sum(test_rmse) / len(test_rmse) / batch_size
    #         #
    #         # test_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
    #         for i in range(3): #3 feature num
    #             output[..., i] = args.scalers[i].inverse_transform(output[..., i])
    #             label[..., i] = args.scalers[i].inverse_transform(label[..., i])
    #         test_mse_loss += test_rmse.item()
    #         test_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
    #         test_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))
    #
    #         half_test_mask_rmse_loss.append(masked_rmse_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))
    #         half_test_mask_mae_loss.append(masked_mae_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))
    #
    #         end_test_mask_rmse_loss.append(masked_rmse_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))
    #         end_test_mask_mae_loss.append(masked_mae_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))
    #
    #         for i in range(3): #3 feature num
    #             output[..., i] = args.scalers[i].transform(output[..., i])
    #             label[..., i] = args.scalers[i].transform(label[..., i])
    #         test_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
    #         half_test_mask_mape_loss.append(masked_mape_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))
    #         end_test_mask_mape_loss.append(masked_mape_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))
    #
    # test_mse_loss = test_mse_loss / test_iters
    # test_mask_rmse_loss = np.mean(test_mask_rmse_loss)
    # test_mask_mape_loss = np.mean(test_mask_mape_loss)
    # test_mask_mae_loss = np.mean(test_mask_mae_loss)
    # half_test_mask_rmse_loss = np.mean(half_test_mask_rmse_loss)
    # half_test_mask_mape_loss = np.mean(half_test_mask_mape_loss)
    # half_test_mask_mae_loss = np.mean(half_test_mask_mae_loss)
    # end_test_mask_rmse_loss = np.mean(end_test_mask_rmse_loss)
    # end_test_mask_mape_loss = np.mean(end_test_mask_mape_loss)
    # end_test_mask_mae_loss = np.mean(end_test_mask_mae_loss)
    #
    # logger.info(f"model: RNN, testing method: {args.mode}, test_RMSE: {test_mse_loss:.4f}")
    # logger.info(f"avg_test_MASK_RMSE: {test_mask_rmse_loss:.4f}, avg_test_MASK_MAE: {test_mask_mae_loss:.4f}, avg_test_MASK_MAPE: {test_mask_mape_loss:.4f}")
    # logger.info(f"half_test_MASK_RMSE:{half_test_mask_rmse_loss:.4f}, half_test_MASK_MAE: {half_test_mask_mae_loss:.4f}, half_test_MASK_MAPE: {half_test_mask_mape_loss:.4f}")
    # logger.info(f"end_test_MASK_RMSE:{end_test_mask_rmse_loss:.4f}, end_test_MASK_MAE: {end_test_mask_mae_loss:.4f}, end_test_MASK_MAPE: {end_test_mask_mape_loss:.4f}, Time: {total_train_time:.4f}")
