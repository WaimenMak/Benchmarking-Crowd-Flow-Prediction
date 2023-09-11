# -*- coding: utf-8 -*-
"""
Created on 11/01/2023 16:12

@Author: mmai
@FileName: metric.py
@Software: PyCharm
"""
import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
def rho_risk(pred, target, timespan, rho):
    """
    timespan: int, number of steps, min:1 max:12
    The input could be any shape.
    """

    window_shape = (pred.shape[0] - timespan + 1, timespan, *pred.shape[1:])
    window_strides_pred = (pred.strides[0], *pred.strides)
    window_strides_target = (target.strides[0], *target.strides)
    # The strides of pred and target are different, so we need to calculate them separately.
    pred = as_strided(pred, shape=window_shape, strides=window_strides_pred) #[N-timespan+1, timespan, steps, nodes, features]
    target = as_strided(target, shape=window_shape, strides=window_strides_target)  # [N-timespan+1, timespan, steps, nodes, features]

    Z_hat = np.sum(pred, axis=1)  # sum on timespan, to get the Z and Z_hat
    Z = np.sum(target, axis=1)  # [N - timespan + 1, ...] [N-timespan+1, steps, nodes, features]

    # Z_hat_rho = np.quantile(Z_hat, q=rho, axis=0)  #actually no need to estimate the quantile.
    # Z_hat_rho = np.quantile(Z, q=rho, axis=0)
    errors = Z_hat - Z # [N, node, features]
    # L = np.abs(errors) * ((rho * (np.sign(Z - Z_hat_rho)+1) + (1-rho) * (np.sign(Z_hat_rho - Z)+1))) #don't use this one, because it's not quantile
    L = np.abs(errors) * ((rho * (np.sign(Z - Z_hat)+1) + (1-rho) * (np.sign(Z_hat - Z)+1))) #use this one, compare pred and gt, rather than quantile


    return L # shape:[N - timespan + 1, ...], the rest dimention remain the same
def weighted_loss(L1, L2, gamma): # L1: rmse, L2: quantile
    return (1-gamma) * L1 + gamma * L2
def weighted_average_loss(pred, target, rho, timespan, mode=0):
    """
    Calculate the weighted average loss, consider different distribution on different nodes.
    pred: all node prediction [N, node, feature], three dimensional
    target: all node ground truth
    """
    min_gamma = 0 # 0.5
    max_gamma = 1 # 0.9
    Q_z = np.quantile(target, q=rho, axis=(0, 2)) # overall 90% quantile of all node, [step, feature]
    Q_znode = np.quantile(target, q=rho, axis=0) # node 90% quantile [step, node, feature]
    diff = Q_znode.transpose([1, 0, 2]) - Q_z     # [node, step, feature]
    gamma = (diff - np.min(diff, axis=0))/(np.max(diff, axis=0) - np.min(diff, axis=0))  #[0~1]
    gamma = min_gamma + gamma * (max_gamma - min_gamma)
    # gamma[gamma<=0.5] = 0.5  # [node, step, feature]
    # gamma[gamma>0.8] = 0.8
    gamma = gamma.transpose(1, 0, 2)
    if mode == 1:
        #rmse
        loss = np.sqrt(np.mean((pred - target)**2, axis=0)) # [step, node, feature]
    if mode == 0:
        #mae
        loss = np.mean(np.abs(pred - target), axis=0) # [step, node, feature]
    rhorisk = rho_risk(pred, target, timespan=timespan, rho=rho)
    qt = np.mean(rhorisk, axis=0)  # [step, node, feature]
    wt = weighted_loss(loss, qt, gamma=gamma) # [step, node, feature]
    # print(wt.shape)
    loss = np.mean(wt)

    return loss, wt, gamma,
def quantile_loss_np(pred, target, rho):
    Z_hat = np.sum(pred, axis=1)  # sum on steps, to get the Z and Z_hat
    Z = np.sum(target, axis=1)  # [Time, 1]
    # Z_hat = pred
    # Z = target
    Z_hat_rho = np.quantile(Z_hat, rho) # Z_rho scalar, Z vector

    errors = Z_hat - Z
    L = np.abs(errors) * ((rho * (np.sign(Z - Z_hat_rho)+1) + (1-rho) * (np.sign(Z_hat_rho - Z)+1))) #use this one
    return L.mean(), Z_hat_rho

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    # preds = preds.reshape(-1, preds.shape[2]*preds.shape[3])
    # labels = labels.reshape(-1, labels.shape[2]*labels.shape[3])
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

def replace_zero(y_true):
    y_true = np.array(y_true)
    mask = y_true < 1e-5
    y_true[mask] = 1e-3  # 将实际值为0的记录替换为1，也可以替换为其他非零值
    return y_true

def masked_mape_np(preds, labels, null_val=np.nan):
    # preds = preds.flatten()
    # labels = labels.flatten()
    # labels = replace_zero(labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-7))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mae_loss():
    def masked_mae(preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels!=null_val)
        mask = mask.float()
        mask /=  torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds-labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    return masked_mae

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
