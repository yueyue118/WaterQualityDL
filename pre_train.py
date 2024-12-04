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
import datetime
import argparse
import collections
import data_process
import utils
import model_config
import model
from train import *
from xpinyin import Pinyin
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

# pre-train
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MultiChannelGSGCN')
    parser.add_argument('-g', '--n_gpu', default=None, type=int, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epochs', default=None, type=int, help='epochs')
    parser.add_argument('-i', '--n_in', default=None, type=int, help='n_in') #修改
    parser.add_argument('-s', '--n_out', default=None, type=int, help='n_out') #修改
    parser.add_argument('-d', '--hidden_size', default=None, type=int, help='hidden_size')
    parser.add_argument('-b', '--batch_size', default=None, type=int, help='batch_size')
    parser.add_argument('-c', '--e_layer', default=None, type=int, help='e_layer')
    parser.add_argument('-n', '--num_heads', default=None, type=int, help='num_heads')
    parser.add_argument('-m', '--mask_ratio', default=None, type=float, help='mask_ratio') #修改
  
    # 传入参数
    args = parser.parse_args(args=['-g', '1', '-e', '300', '-i', '8', '-s', '8', '-d', '64', '-b', '32', '-c', '3', '-n', '8', '-m', '0.7'])
    print(args)
    
# 加载参数
config = model.ModelConfig(args, model_config)

#加载数据集
p = Pinyin()
name = '珠江'
Eng_name = p.get_pinyin(name, '')

dataset_files = data_process.get_dataset_files(os.path.join(config.base_dir, config.dataset_dir), name)
config.WQ_data_name = dataset_files

#加载数据集
train_loader, test_loader, scaler = utils.pre_train_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
                                                            dataset_files=config.WQ_data_name, sequence_length=config.n_in,
                                                            num_feat=config.input_dim, batch_size=config.batch_size,
                                                            train_split=config.train_ratio, mask_ratio=config.mask_ratio)

for x, y in train_loader:
    x_train = x
    y_train = y
    
for x, y in test_loader:
    x_test = x
    y_test = y

# get number of iterations per epoch for progress bar
num_train_sample = x_train.shape[0]
num_test_sample = x_test.shape[0]
num_train_iter_per_epoch = math.ceil(num_train_sample / config.batch_size)
num_test_iter_per_epoch = math.ceil(num_test_sample / config.batch_size)

# (batch_size, patch_length, num_nodes, parameters, patch_size)
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
model = model.MultiTransformer(num_heads=config.num_heads, e_layer=config.e_layer, hidden_size=config.hidden_size,
                            num_stations=len(dataset_files), num_feat=config.input_dim,
                            seq_len=config.n_in, pred_len=config.n_out)
loss_func = nn.MSELoss()
# 可训练模型
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.base_lr, weight_decay=0.0, eps=config.epsilon, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)

epoch_losses, epoch_train_maes, epoch_train_rmse, epoch_train_mapes, val_losses, val_maes, val_rmses, val_mapes, average_training_time = pre_train(model, loss_func, optimizer, config, train_loader=train_loader,
                                                                          val_loader=test_loader, lr_scheduler=lr_scheduler,
                                                                          len_epoch=num_train_iter_per_epoch, val_len_epoch=num_test_iter_per_epoch)

config.model_pkl_filename = f'./src/model/pretrain_{Eng_name}_{config.mask_ratio}.pth'
torch.save(model, config.model_pkl_filename)
results = pd.DataFrame({"train_loss": epoch_losses, "val_loss": val_losses})
results.to_csv(f"F:/河道水质预测/src/saved/pretrain_{Eng_name}_{config.mask_ratio}.csv")