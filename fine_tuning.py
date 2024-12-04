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
from train import *
import test
from xpinyin import Pinyin
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

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
    args = parser.parse_args(args=['-g', '1', '-e', '100', '-i', '8', '-s', '1', '-d', '64', '-b', '32', '-c', '3', '-n', '8', '-m', '0.5'])
    print(args)
    
# 加载参数
config = model.ModelConfig(args, model_config)
config.base_lr = 1e-2
config.lr_milestones = [40, 60, 80]
config.lr_decay_ratio = 0.5
p = Pinyin()
name = '黄河'
Eng_name = p.get_pinyin(name, '')

dataset_files = data_process.get_dataset_files(os.path.join(config.base_dir, config.dataset_dir), name)
config.W_data_name = '黑龙江_148.csv'
config.M_data_name = 'Meteorology_黑龙江_148.csv'

#加载数据集
# W_train_loader, W_test_loader, M_train_loader, M_test_loader, scaler = utils.fine_tune_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
#                                                             W_dataset_file=config.W_data_name, M_dataset_file=config.M_data_name, sequence_length=config.n_in,
#                                                             pre_len=config.n_out, num_feat=config.input_dim, M_num_feat=config.feature_dim,
#                                                             batch_size=config.batch_size, train_split=config.train_ratio)

#不同比例训练集
W_train_loader_1, W_test_loader_1, M_train_loader_1, M_test_loader_1, scaler_1 = utils.fine_tune_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
                                                            W_dataset_file=config.W_data_name, M_dataset_file=config.M_data_name, sequence_length=config.n_in,
                                                            pre_len=config.n_out, num_feat=config.input_dim, M_num_feat=config.feature_dim,
                                                            batch_size=config.batch_size, train_split=config.train_ratio)

config.train_ratio = 0.16

W_train_loader_2, W_test_loader_2, M_train_loader_2, M_test_loader_2, scaler_2 = utils.fine_tune_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
                                                            W_dataset_file=config.W_data_name, M_dataset_file=config.M_data_name, sequence_length=config.n_in,
                                                            pre_len=config.n_out, num_feat=config.input_dim, M_num_feat=config.feature_dim,
                                                            batch_size=config.batch_size, train_split=config.train_ratio)

W_train_loader = W_train_loader_2
W_test_loader = W_test_loader_1
M_train_loader = M_train_loader_2
M_test_loader = M_test_loader_1
scaler = scaler_1


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
model = model.Prompt_MultiTransformer(num_heads=config.num_heads, e_layer=config.e_layer, hidden_size=config.hidden_size, prompt_num=1,
                            num_stations=len(dataset_files), num_feat=config.input_dim,
                            seq_len=config.n_in, pred_len=config.n_out)

# 加载并冻结参数
pre_train_weight_path = f'F:/河道水质预测/src/model/pretrain_{Eng_name}_{config.mask_ratio}.pth'
ckpt = torch.load(pre_train_weight_path).state_dict()
model_dict = model.state_dict()
pre_train_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
model_dict.update(pre_train_dict)
model.load_state_dict(model_dict, strict=False)
frozen_layer = ['mlp.weight', 'mlp.bias']
for name, para in model.named_parameters():
    if name in pre_train_dict and name not in frozen_layer:
        para.requires_grad=False


loss_func = nn.MSELoss()
# 可训练模型
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.base_lr, weight_decay=0.0, eps=config.epsilon, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)

start_time = time.time()

epoch_losses, epoch_train_maes, epoch_train_rmse, epoch_train_mapes, val_losses, val_maes, val_rmses, val_mapes, average_training_time = fine_tuning(model, loss_func, optimizer, config,
                                                                        W_train_loader=W_train_loader, M_train_loader=M_train_loader,
                                                                        W_val_loader=W_test_loader, M_val_loader=M_test_loader, lr_scheduler=lr_scheduler,
                                                                          len_epoch=num_train_iter_per_epoch, val_len_epoch=num_test_iter_per_epoch)

end_time = time.time()
print("训练耗时：{:.2f}秒".format(end_time - start_time))

# config.model_pkl_filename = f"./src/model/Frozen_{Eng_name}_{config.mask_ratio}_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.pth"
config.model_pkl_filename = f"./src/model/Frozen_{Eng_name}_Transformer_0.2_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.pth"

torch.save(model, config.model_pkl_filename)
results = pd.DataFrame({"train_loss": epoch_losses, "val_loss": val_losses})
# results.to_csv(f"F:/河道水质预测/src/saved/Frozen_{Eng_name}_{config.mask_ratio}_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")
results.to_csv(f"F:/河道水质预测/src/saved/Frozen_{Eng_name}_Transformer_0.2_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")

outputs, targets = test.predict(model, config, scaler, W_test_loader, M_test_loader)
parameters = ['CODMn', 'DO', 'NH4N', 'pH']
data = {}
for i, param in enumerate(parameters):
    data[f'output_{param}'] = outputs[i, :]  # output的列
    data[f'target_{param}'] = targets[i, :]  # target的列

results = pd.DataFrame(data)
# results.to_csv(f"F:/河道水质预测/src/saved/results/result_Frozen_{Eng_name}_{config.mask_ratio}_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")
results.to_csv(f"F:/河道水质预测/src/saved/results/result_Frozen_{Eng_name}_Transformer_0.2_{p.get_pinyin(os.path.splitext(config.W_data_name)[0], '')}.csv")
