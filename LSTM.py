import pickle as pkl
import argparse
import torch
import pandas as pd
import numpy as np
import math
import os
import sys
import numpy.linalg as la
import scipy.sparse as sp
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from scipy.sparse.linalg import eigs
import json
import yaml
import datetime
import argparse
import collections
import data_process
import utils
import model_config
import model
import train
import test
from xpinyin import Pinyin
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_weather):
        # 初始化LSTM的隐藏状态和细胞状态
        x_combined = torch.cat((x, x_weather), dim=-1)
        h0 = torch.zeros(self.num_layers, x_combined.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x_combined.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x_combined, (h0, c0))
        
        # 取出LSTM最后一个时间步的输出，输入全连接层
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out

def lstm_train(model, loss_func, optimizer, config, W_train_loader, M_train_loader, W_val_loader, M_val_loader, lr_scheduler, len_epoch, val_len_epoch):
    n_in = config.n_in
    n_out = config.n_out

    device = config.device
    device_ids = config.device_ids
    if len(config.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model = model.to(device)

    n_epochs = config.epochs

    base_dir = config.base_dir
    
    iter_losses = np.zeros(n_epochs * len_epoch)
    epoch_losses = np.zeros(n_epochs)
    epoch_train_maes = np.zeros(n_epochs)
    epoch_train_rmses = np.zeros(n_epochs)
    epoch_train_mapes = np.zeros(n_epochs)
    val_losses = np.zeros(int(np.ceil(n_epochs * 1./1)))
    val_maes = np.zeros(int(np.ceil(n_epochs * 1./1)))
    val_rmses = np.zeros(int(np.ceil(n_epochs * 1./1)))
    val_mapes = np.zeros(int(np.ceil(n_epochs * 1./1)))
    max_grad_norm = config.max_grad_norm
    # loss function
    loss_func = loss_func.to(device)
    print(f"Iterations train per epoch:{len_epoch:d}.")
    print(f"Using computation device: {device}.")

    n_iter = 0
    training_time = 0
    for e_i in range(n_epochs):
        start_time = time.time()
        total_loss = 0
        total_acc_rmse = 0
        total_acc_mae = 0
        total_acc_mape = 0

        for (batch_idx, (data, target)), (m_batch_idx, (m_data, m_target)) in zip(enumerate(W_train_loader), enumerate(M_train_loader)):
            data = data.to(device)
            m_data = m_data.to(device)
            optimizer.zero_grad()
            # TODO: need edit
            model.train()
            output = model(data, m_data)
            output = output.cpu()
            loss = loss_func(output, target.squeeze(1))
            train_iteration = e_i * len_epoch + batch_idx
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _ = train.fine_tune_eval_metrics(output.detach().cpu().numpy(), target.squeeze(1).numpy())
            total_acc_mae += acc_mae
            total_acc_rmse += acc_rmse
            total_acc_mape += acc_mape

        lr_scheduler.step()
        train_epoch_time = time.time() - start_time
        epoch_losses[e_i] = total_loss / len_epoch
        epoch_train_maes[e_i] = total_acc_mae / len_epoch
        epoch_train_rmses[e_i] = total_acc_rmse / len_epoch
        epoch_train_mapes[e_i] = total_acc_mape / len_epoch

        training_time += train_epoch_time
        if e_i % 10 == 0:
            val_loss, val_mae, val_rmse, val_mape = lstm_test(model, loss_func, optimizer, config,
                                                        W_test_loader=W_val_loader, M_test_loader=M_val_loader, val_len_epoch=val_len_epoch)
            print(f"train_epoch_time: {train_epoch_time}")
            # TODO: make this MSE and make it work for multiple inputs
            # val_loss = np.mean(np.abs(y_val_pred - y_val_true))
            print(f"Epoch: {e_i}, train_loss:{epoch_losses[e_i]}, train_rmse:{epoch_train_rmses[e_i]}, train_mae:{epoch_train_maes[e_i]},  train_mape:{epoch_train_mapes[e_i]}")
            print(f"Epoch: {e_i}, val_loss: {val_loss}, val_rmse: {val_rmse}, val_mae: {val_mae}, val_mape: {val_mape}")
            val_iteration = int(np.ceil(e_i * 1. / 1))
                        
            val_losses[val_iteration] = val_loss
            val_maes[val_iteration] = val_mae
            val_rmses[val_iteration] = val_rmse
            val_mapes[val_iteration] = val_mape

        # There is a chance that the training loss will explode, the temporary workaround
        # is to restart from the last saved model before the explosion, or to decrease
        # the learning rate earlier in the learning rate schedule.
        if epoch_losses[e_i] > 1e5:
            print('Gradient explosion detected. Ending...')
            break

    average_training_time = training_time / n_epochs
    print("Average training time: {:.4f}s".format(average_training_time))
    return epoch_losses, epoch_train_maes, epoch_train_rmses, epoch_train_mapes, val_losses, val_maes, val_rmses, val_mapes, average_training_time

def lstm_test(model, loss_func, optimizer, config, W_test_loader, M_test_loader, val_len_epoch):
    model.eval()
    total_val_loss = 0
    #total_val_metrics = np.zeros(len(self.metrics))
    total_acc_rmse = 0
    total_acc_mae = 0
    total_acc_mape = 0
    device = config.device

    with torch.no_grad():
        for (batch_idx, (data, target)), (m_batch_idx, (m_data, m_target)) in zip(enumerate(W_test_loader), enumerate(M_test_loader)):
            data = data.to(device)
            m_data = m_data.to(device)
            output = model(data, m_data)
            output = output.cpu()
            loss = loss_func(output, target.squeeze(1))
            total_val_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _ = train.fine_tune_eval_metrics(output.detach().numpy(), target.squeeze(1).numpy())
            total_acc_mae += acc_mae
            total_acc_rmse += acc_rmse
            total_acc_mape += acc_mape

        val_loss = total_val_loss / val_len_epoch
        val_mae = total_acc_mae / val_len_epoch
        val_rmse = total_acc_rmse / val_len_epoch
        val_mape = total_acc_mape / val_len_epoch

    return val_loss, val_mae, val_rmse, val_mape


def calculate_nse(observed, predicted):
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    return 1 - (numerator / denominator)

# pre-train
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MultiChannelGSGCN')
    parser.add_argument('-g', '--n_gpu', default=None, type=int, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epochs', default=None, type=int, help='epochs')
    parser.add_argument('-i', '--n_in', default=None, type=int, help='n_in')
    parser.add_argument('-s', '--n_out', default=None, type=int, help='n_out')
    parser.add_argument('-d', '--hidden_size', default=None, type=int, help='hidden_size')
    parser.add_argument('-b', '--batch_size', default=None, type=int, help='batch_size')
    parser.add_argument('-c', '--e_layer', default=None, type=int, help='e_layer')
    parser.add_argument('-n', '--num_heads', default=None, type=int, help='num_heads')
    parser.add_argument('-m', '--mask_ratio', default=None, type=float, help='mask_ratio')
  
    # 传入参数
    args = parser.parse_args(args=['-g', '1', '-e', '300', '-i', '8', '-s', '1', '-d', '64', '-b', '32', '-c', '2', '-n', '8', '-m', '0.5'])
    print(args)
    
# 加载参数
config = model.ModelConfig(args, model_config)
config.base_lr = 1e-1
config.lr_milestones = [40, 80, 120, 160, 180, 200, 220, 240, 260, 280]
config.lr_decay_ratio = 0.5
p = Pinyin()
name = '珠江'
Eng_name = p.get_pinyin(name, '')

dataset_files = data_process.get_dataset_files(os.path.join(config.base_dir, config.dataset_dir), name)
config.W_data_name = '其他_149.csv'
config.M_data_name = 'Meteorology_其他_149.csv'

#加载数据集
W_train_loader, W_test_loader, M_train_loader, M_test_loader, scaler = utils.fine_tune_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
                                                            W_dataset_file=config.W_data_name, M_dataset_file=config.M_data_name, sequence_length=config.n_in,
                                                            pre_len=config.n_out, num_feat=config.input_dim, M_num_feat=config.feature_dim,
                                                            batch_size=config.batch_size, train_split=config.train_ratio)

for x, y in W_train_loader:
    x_train = x
    y_train = y
    break
    
for x, y in W_test_loader:
    x_test = x
    y_test = y
    break

# get number of iterations per epoch for progress bar
num_train_sample = x_train.shape[0]
num_test_sample = x_test.shape[0]
num_train_iter_per_epoch = math.ceil(num_train_sample / config.batch_size)
num_test_iter_per_epoch = math.ceil(num_test_sample / config.batch_size)

print(f"data:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

print(f"model architecture:")
print(f"model_name: {config.model_name}")
print(f"e_layer: {config.e_layer}")
print(f"num_heads: {config.num_heads}, hidden_size: {config.hidden_size}")
print(f"n_in: {config.n_in}, n_out: {config.n_out}, epochs: {config.epochs}")
print(f"gpu: {config.n_gpu}, batch_size: {config.batch_size}")
print(f"input_dim: {config.input_dim}, output_dim: {config.output_dim}")

# 加载模型
model = LSTM(config.input_dim+config.feature_dim, config.hidden_size, config.input_dim, config.e_layer)

loss_func = nn.MSELoss()
# 可训练模型
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.base_lr, weight_decay=0.0, eps=config.epsilon, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)

start_time = time.time()

epoch_losses, epoch_train_maes, epoch_train_rmse, epoch_train_mapes, val_losses, val_maes, val_rmses, val_mapes, average_training_time = lstm_train(model, loss_func, optimizer, config,
                                                                        W_train_loader=W_train_loader, M_train_loader=M_train_loader,
                                                                        W_val_loader=W_test_loader, M_val_loader=M_test_loader, lr_scheduler=lr_scheduler,
                                                                          len_epoch=num_train_iter_per_epoch, val_len_epoch=num_test_iter_per_epoch)

end_time = time.time()
print("训练耗时：{:.2f}秒".format(end_time - start_time))

config.model_pkl_filename = f"./src/model/LSTM_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.pth"
torch.save(model, config.model_pkl_filename)
results = pd.DataFrame({"train_loss": epoch_losses, "val_loss": val_losses})
results.to_csv(f"F:/河道水质预测/src/saved/LSTM_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")

outputs, targets = test.predict(model, config, scaler, W_test_loader, M_test_loader)
parameters = ['CODMn', 'DO', 'NH4N', 'pH']
data = {}
for i, param in enumerate(parameters):
    data[f'output_{param}'] = outputs[i, :]  # output的列
    data[f'target_{param}'] = targets[i, :]  # target的列

results = pd.DataFrame(data)
results.to_csv(f"F:/河道水质预测/src/saved/results/result_LSTM_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")