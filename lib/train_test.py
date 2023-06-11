# -*- coding: utf-8 -*-
# @Time    : 12/04/2023 15:10
# @Author  : mmai
# @FileName: train_test
# @Software: PyCharm
import time
import os
import torch
import numpy as np
import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter
from lib.utils import _compute_sampling_threshold
from lib.metric import masked_rmse_np, masked_mape_np, masked_mae_np, masked_mae_loss

def sum_step_loss(args):
    def sum_step(output, label):
        """
        calculate the loss step by step
        """
        loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
        loss = sum(loss_sup_seq) / len(loss_sup_seq) / args.batch_size
        return loss
    return sum_step

class Trainer():
    def __init__(self, model, args, logger):
        self.model = model
        self.args = args
        self.logger = logger
        self.step = 20 # for curriculum learning, train every step of iteration, increase the learning level
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = args.seq_len
        self.cl = args.cl
        if args.loss_func == "sum steps":
            self.loss = sum_step_loss()
        elif args.loss_func == "masked mae":
            self.loss = masked_mae_loss()
        else:
            self.loss = torch.nn.MSELoss(reduction="mean")
    def train(self):
        args = self.args
        log_dir = "./result/"+args.filename+"_logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        model = self.model
        logger = self.logger
        model.to(args.device)
        best_val = float('inf')
        train_list, val_list, test_list = [], [], []
        model.to(args.device)
        start_time = time.time()  # record the start time of each epoch
        for epoch in range(args.len_epoch):
            model.train()
            train_rmse_loss = 0
            # training_time += train_epoch_time
            '''
            code from dcrnn_trainer.py, _train_epoch()
            '''
            # total_loss = 0
            # total_metrics = np.zeros(len(metrics))
            for batch_idx, (data, target) in enumerate(args.data_loader.get_iterator()):
                data = torch.FloatTensor(data)
                data = data[:, -args.step:, :, :]  # defining input sequence length
                # data = data[..., :1]
                target = torch.FloatTensor(target)
                label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
                data, target = data.to(args.device), target.to(args.device)

                args.optimizer.zero_grad()

                #data [bc, seq, node, feature]
                global_step = (epoch+1 - 1) * args.len_epoch + batch_idx
                teacher_forcing_ratio = _compute_sampling_threshold(global_step, args.cl_decay_steps)

                #data [bc, seq, node, feature]
                # for i in range(num_nodes):
                # output = model(data.permute(0, 2, 1, 3).reshape([-1, seq_len, output_dim]), target.permute(0, 2, 1, 3).reshape(
                #     [-1, seq_len, output_dim]), teacher_forcing_ratio)
                # data [bc, seq_len, feature 105]
                if args.filename == 'mtgnn':
                    if batch_idx%args.step_size2==0:
                        perm = np.random.permutation(range(args.num_nodes))
                    num_sub = int(args.num_nodes/args.num_split)
                    for j in range(args.num_split):
                        if j != args.num_split-1:
                            id = perm[j * num_sub:(j + 1) * num_sub]
                        else:
                            id = perm[j * num_sub:]
                        id = torch.tensor(id).to(args.device)
                        tx = data[:, :, id, :]
                        ty = label[:, :, id, :]
                        output = model(tx.transpose(3, 1), idx=id)
                        output = output.transpose(1,3)
                        label = ty
                else:
                    # output should be [bc, seq_len, num_nodes, features]
                    output = model(data, target, teacher_forcing_ratio)
                # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
                # loss_sup_seq = [torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2) for step_t in range(12)]  #training loss function
                train_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(args.seq_len)] # training metric for each step

                train_rmse = sum(train_rmse) / len(train_rmse) / args.batch_size
                #curriculum learning, code from MTGNN trainer.py
                if self.iter%self.step==0 and self.task_level<=self.seq_out_len: # seq_out_len 12
                    self.task_level +=1
                if self.cl:
                    loss = self.loss(output[:, :self.task_level, :, :], label[:, :self.task_level, :, :])
                else:
                    loss = self.loss(output, label)  # loss is self-defined, need cpu input

                loss.backward()
                # add max grad clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                args.optimizer.step()
                self.iter += 1

                # writer.set_step((epoch - 1) * len_epoch + batch_idx)
                # writer.add_scalar('loss', loss.item())
                # total_loss += loss.item()
                # train_mse_loss += loss.item()
                train_rmse_loss += train_rmse.item()  #metric  sum of each node and each iteration


                if batch_idx == args.len_epoch:
                    break

            train_rmse_loss = train_rmse_loss / args.train_iters
            # print(f"Epoch: {epoch}, train_RMSE: {train_rmse_loss}")
            writer.add_scalar("Loss/train", loss.item(), global_step=epoch)
            logger.info(f"Epoch [{epoch+1}/{args.len_epoch}], train_RMSE: {train_rmse_loss:.4f}")



            # validation
            if (epoch+1) % 5 == 0:
                val_mse_loss = []
                val_mask_rmse_loss = []
                val_mask_mae_loss = []
                val_mask_mape_loss = []
                # validation
                model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(args.val_dataloader.get_iterator()):
                        data = torch.FloatTensor(data)
                        data = data[:, -args.step:,:,:]
                        target = torch.FloatTensor(target)
                        label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
                        data, target = data.to(args.device), target.to(args.device)


                        output = model(data, target, teacher_forcing_ratio=0)
                        # output = output.reshape([args.batch_size, args.num_nodes, args.seq_len, -1]).permute(0, 2, 1, 3)
                        val_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(args.seq_len)]   # 12 graphs
                        val_rmse = sum(val_rmse) / len(val_rmse) / args.batch_size

                        val_mse_loss.append(val_rmse.item())
                        val_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
                        val_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
                        val_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))

                # val_mse_loss = val_mse_loss / args.val_iters
                val_mse_loss = np.mean(val_mse_loss)
                for i in range(args.features): #3 feature num
                    output[..., i] = args.scalers[i].inverse_transform(output[..., i])
                    label[..., i] = args.scalers[i].inverse_transform(label[..., i])
                val_mask_rmse_loss = np.mean(val_mask_rmse_loss)
                val_mask_mape_loss = np.mean(val_mask_mape_loss)
                val_mask_mae_loss = np.mean(val_mask_mae_loss)
                # print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
                logger.info(f"Epoch [{epoch+1}/{args.len_epoch}], val_RMSE: {val_mse_loss:.4f}, val_MASK_RMSE: {val_mask_rmse_loss:.4f}, val_MASK_MAE: {val_mask_mae_loss:.4f}, val_MASK_MAPE: {val_mask_mape_loss:.4f}")
                # print(f"Epoch: {epoch}, val_RMSE: {val_mse_loss}")
                # logger.info(f"Epoch [{epoch+1}/{args.len_epoch}], val_RMSE: {val_mse_loss:.4f}")
                train_list.append(train_rmse_loss)
                val_list.append(val_mse_loss)
                if args.mode == "in-sample":
                    np.save("./result/"+args.filename+"_train_loss.npy", train_list)
                    np.save("./result/"+args.filename+"_val_loss.npy", val_list)
                    if val_mse_loss < best_val:
                        best_val = val_mse_loss
                        torch.save(model.state_dict(), "./result/"+args.filename+".pt")
                else:
                    np.save("./result/"+args.filename+"_train_loss_ood.npy", train_list)
                    np.save("./result/"+args.filename+"_val_loss_ood.npy", val_list)
                    if val_mse_loss < best_val:
                        best_val = val_mse_loss
                        torch.save(model.state_dict(), "./result/"+args.filename+"_ood.pt")
                # if early_stopper.early_stop(val_mse_loss):
                #     # break
                #     pass

        end_time = time.time()
        total_train_time = end_time - start_time
        return total_train_time
    # logger.info(f"testing method: {args.mode}, test_RMSE: {test_mse_loss:.4f},  Time: {total_train_time:.4f}")

    def test(self, total_train_time):
        test_mse_loss = []
        args = self.args
        model = self.model
        test_mask_rmse_loss = []
        test_mask_mae_loss = []
        test_mask_mape_loss = []
        half_test_mask_rmse_loss = []
        half_test_mask_mae_loss = []
        half_test_mask_mape_loss = []
        end_test_mask_rmse_loss = []
        end_test_mask_mae_loss = []
        end_test_mask_mape_loss = []
        if args.mode == "in-sample":
            model.load_state_dict(torch.load("./result/"+args.filename+".pt"))
        elif args.mode == "ood":
            model.load_state_dict(torch.load("./result/"+args.filename+"_ood.pt"))
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(args.test_dataloader.get_iterator()):
                data = torch.FloatTensor(data)
                # data = data[..., :1]
                target = torch.FloatTensor(target)
                label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
                data, target = data.to(args.device), target.to(args.device)

                # output = model(data[:, :, i, :], torch.zeros(target[:, :, i, :].size()), 0)
                # output = torch.transpose(output.view(12, model.batch_size, model.num_nodes,
                #                              model._output_dim), 0, 1)  # back to (50, 12, 207, 1)
                output = model(data, target, teacher_forcing_ratio=0)
                # output = output.reshape([args.batch_size, args.num_nodes, args.seq_len, -1]).permute(0, 2, 1, 3)
                # output = output.reshape([batch_size, num_nodes, seq_len, output_dim]).permute(0, 2, 1, 3)
                test_rmse = [torch.sum(torch.sqrt(torch.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, dim=(1,2)))) for step_t in range(12)]
                test_rmse = sum(test_rmse) / len(test_rmse) / args.batch_size

                # test_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
                for i in range(3): #3 feature num
                    output[..., i] = args.scalers[i].inverse_transform(output[..., i])
                    label[..., i] = args.scalers[i].inverse_transform(label[..., i])
                # test_mse_loss += test_rmse.item()
                test_mse_loss.append(test_rmse.item())
                test_mask_rmse_loss.append(masked_rmse_np(output.numpy(), label.numpy()))
                test_mask_mae_loss.append(masked_mae_np(output.numpy(), label.numpy()))

                half_test_mask_rmse_loss.append(masked_rmse_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))
                half_test_mask_mae_loss.append(masked_mae_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))

                end_test_mask_rmse_loss.append(masked_rmse_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))
                end_test_mask_mae_loss.append(masked_mae_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))

                for i in range(3): #3 feature num
                    output[..., i] = args.scalers[i].transform(output[..., i])
                    label[..., i] = args.scalers[i].transform(label[..., i])
                test_mask_mape_loss.append(masked_mape_np(output.numpy(), label.numpy()))
                half_test_mask_mape_loss.append(masked_mape_np(output.numpy()[:,5,:,:], label.numpy()[:,5,:,:]))
                end_test_mask_mape_loss.append(masked_mape_np(output.numpy()[:,11,:,:], label.numpy()[:,11,:,:]))

        # test_mse_loss = test_mse_loss / args.test_iters
        test_mse_loss = np.mean(test_mse_loss)
        test_mask_rmse_loss = np.mean(test_mask_rmse_loss)
        test_mask_mape_loss = np.mean(test_mask_mape_loss)
        test_mask_mae_loss = np.mean(test_mask_mae_loss)
        half_test_mask_rmse_loss = np.mean(half_test_mask_rmse_loss)
        half_test_mask_mape_loss = np.mean(half_test_mask_mape_loss)
        half_test_mask_mae_loss = np.mean(half_test_mask_mae_loss)
        end_test_mask_rmse_loss = np.mean(end_test_mask_rmse_loss)
        end_test_mask_mape_loss = np.mean(end_test_mask_mape_loss)
        end_test_mask_mae_loss = np.mean(end_test_mask_mae_loss)

        self.logger.info(f"model: {args.filename}, testing method: {args.mode}, curriculum: {args.cl}, test_RMSE: {test_mse_loss:.4f}, input seq len:{args.step}")
        self.logger.info(f"half_test_MASK_RMSE:{half_test_mask_rmse_loss:.4f}, half_test_MASK_MAE: {half_test_mask_mae_loss:.4f}, half_test_MASK_MAPE: {half_test_mask_mape_loss:.4f}")
        self.logger.info(f"end_test_MASK_RMSE:{end_test_mask_rmse_loss:.4f}, end_test_MASK_MAE: {end_test_mask_mae_loss:.4f}, end_test_MASK_MAPE: {end_test_mask_mape_loss:.4f}")
        self.logger.info(f"avg_test_MASK_RMSE: {test_mask_rmse_loss:.4f}, avg_test_MASK_MAE: {test_mask_mae_loss:.4f}, avg_test_MASK_MAPE: {test_mask_mape_loss:.4f}, Time: {total_train_time:.4f}")

def test_ml(data, loaded_model, args, logger, total_train_time):
    test_mse_loss = []
    test_mask_rmse_loss = []
    test_mask_mae_loss = []
    test_mask_mape_loss = []
    half_test_mask_rmse_loss = []
    half_test_mask_mae_loss = []
    half_test_mask_mape_loss = []
    end_test_mask_rmse_loss = []
    end_test_mask_mae_loss = []
    end_test_mask_mape_loss = []

    test_dataloader = data["test_loader"]
    for batch_idx, (input, org_target) in enumerate(test_dataloader.get_iterator()):
        # for i in range(args.features): #3 feature num
        #     input[..., i] = data["scalers"][i].inverse_transform(input[..., i]) # turn to original data

        # target = org_target.reshape([-1, args.num_nodes * args.features * args.seq_len])
        # label = target[..., :model._output_dim]  # (..., 1)  supposed to be numpy array
        label = org_target
        if isinstance(loaded_model, list):
            outputs = []
            for j, model in enumerate(loaded_model):
                if args.filename in ("xgboost", "xgboost v1"):
                    # import xgboost as xgb
                    output = model.predict(xgb.DMatrix(input[:,-args.step:,j,:].reshape([-1, args.step * args.features]))) # args.steps is the lags, seq_len is the predicted future steps
                else:
                    output = model.predict(input[:,-args.step:,j,:].reshape([-1, args.step * args.features])) # output [bc, seq_len, features]
                output = output.reshape([args.batch_size, args.seq_len, args.features])
                outputs.append(output)

            output = np.stack(outputs, axis=2)
        else:
            if args.filename == "rnn":
                input = input[:,-args.step:,:,:].reshape([-1, args.step, args.num_nodes * args.features])
            else:
                input = input[:,-args.step:,:,:].reshape([-1, args.num_nodes * args.features * args.step])  # args.steps is the lags, seq_len is the predicted future steps

            if args.filename in ("xgboost", "xgboost v1"): # testing xgboost
                # import xgboost as xgb
                output = loaded_model.predict(xgb.DMatrix(input))
            else:
                output = loaded_model.predict(input)

        output = output.reshape([args.batch_size, args.seq_len, args.num_nodes, args.features])

        for i in range(args.features):
            output[..., i] = data["scalers"][i].inverse_transform(output[..., i])
            label[..., i] = data["scalers"][i].inverse_transform(label[..., i])   #normalize
        test_rmse = [np.sum(np.sqrt(np.sum((output[:, step_t, :, :] - label[:, step_t, :, :]) ** 2, axis=(1,2)))) for step_t in range(12)]
        test_rmse = sum(test_rmse) / len(test_rmse) / args.batch_size


        test_mse_loss.append(test_rmse.item())
        test_mask_rmse_loss.append(masked_rmse_np(output, label)) # avg
        half_test_mask_rmse_loss.append(masked_rmse_np(output[:,5,:,:], label[:,5,:,:])) # half
        end_test_mask_rmse_loss.append(masked_rmse_np(output[:,11,:,:], label[:,11,:,:])) # end
        test_mask_mae_loss.append(masked_mae_np(output, label))
        half_test_mask_mae_loss.append(masked_mae_np(output[:,5,:,:], label[:,5,:,:]))
        end_test_mask_mae_loss.append(masked_mae_np(output[:,11,:,:], label[:,11,:,:]))
        for i in range(args.features):
            output[..., i] = data["scalers"][i].transform(output[..., i])   #normalize
            label[..., i] = data["scalers"][i].transform(label[..., i])   #normalize
        test_mask_mape_loss.append(masked_mape_np(output, label))
        half_test_mask_mape_loss.append(masked_mape_np(output[:,5,:,:], label[:,5,:,:]))
        end_test_mask_mape_loss.append(masked_mape_np(output[:,11,:,:], label[:,11,:,:]))


    # test_mse_loss = test_mse_loss / test_iters
    test_mse_loss = np.mean(test_mse_loss)
    test_mask_rmse_loss = np.mean(test_mask_rmse_loss)
    test_mask_mape_loss = np.mean(test_mask_mape_loss)
    test_mask_mae_loss = np.mean(test_mask_mae_loss)
    half_test_mask_rmse_loss = np.mean(half_test_mask_rmse_loss)
    half_test_mask_mape_loss = np.mean(half_test_mask_mape_loss)
    half_test_mask_mae_loss = np.mean(half_test_mask_mae_loss)
    end_test_mask_rmse_loss = np.mean(end_test_mask_rmse_loss)
    end_test_mask_mape_loss = np.mean(end_test_mask_mape_loss)
    end_test_mask_mae_loss = np.mean(end_test_mask_mae_loss)
    logger.info(f"model: {args.filename}, testing method: {args.mode}, test_RMSE: {test_mse_loss:.4f}, input seq len: {args.step}")
    logger.info(f"half_test_MASK_RMSE:{half_test_mask_rmse_loss:.4f}, half_test_MASK_MAE: {half_test_mask_mae_loss:.4f}, half_test_MASK_MAPE: {half_test_mask_mape_loss:.4f}")
    logger.info(f"end_test_MASK_RMSE:{end_test_mask_rmse_loss:.4f}, end_test_MASK_MAE: {end_test_mask_mae_loss:.4f}, end_test_MASK_MAPE: {end_test_mask_mape_loss:.4f}")
    logger.info(f"avg_test_MASK_RMSE: {test_mask_rmse_loss:.4f}, avg_test_MASK_MAE: {test_mask_mae_loss:.4f}, avg_test_MASK_MAPE: {test_mask_mape_loss:.4f}, Time: {total_train_time:.4f}")