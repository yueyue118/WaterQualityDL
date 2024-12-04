import os
import pandas as pd
import numpy as np
import torch
import model_config
import data_process
import mask_strategy
import pickle
from torch.utils.data import DataLoader, Dataset, TensorDataset

def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    gpu_num = torch.cuda.device_count()
    if n_gpu_use > 0 and gpu_num == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > gpu_num:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = gpu_num
    if n_gpu_use > 0:
        device = torch.device('cuda:0')
        list_ids = list(range(n_gpu_use))
    else:
        device = torch.device('cpu')
        list_ids = []
    return device, list_ids

def inverse_scaler(scaler, null_val):
    def inverse(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return preds, labels

    return inverse

class CustomDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)

def apply_masking(dataset, random_ratio, param_ratio, station_ratio, temporal_ratio):
    new_data_x = []
    new_data_y = []
    
    for i in range(len(dataset)):
        x, y = dataset[i]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.unsqueeze(0)  # 添加batch维度以适配掩码函数
        # 依次应用四种掩码
        masked_x_random = mask_strategy.random_masking(x, random_ratio)
        masked_x_param = mask_strategy.parameter_masking(x, param_ratio)
        masked_x_station = mask_strategy.station_masking(x, station_ratio)
        masked_x_temporal = mask_strategy.temporal_masking(x, temporal_ratio)
        
        # 拼接原始数据和四种掩码后的数据 (batch, seq_len, num_feat, num_stations) => (5*batch, seq_len, num_feat, num_stations)
        combined_x = torch.cat([x, masked_x_random, masked_x_param, masked_x_station, masked_x_temporal], dim=0)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        y = y.unsqueeze(0)
        combined_y = y.repeat(5, 1, 1, 1)
        
        new_data_x.append(combined_x)
        new_data_y.append(combined_y)
    
    # 将列表中的数据合并为单个张量
    new_data_x = torch.cat(new_data_x, dim=0)
    new_data_y = torch.cat(new_data_y, dim=0)
    
    return CustomDataset(new_data_x, new_data_y)

def pre_train_load_dataset(dataset_dir, dataset_files, sequence_length, num_feat, batch_size, train_split, mask_ratio):
    df_list, scaler = data_process.pre_train_read_dataset(dataset_dir, dataset_files)
    
    train_dataset = data_process.pre_train_process_data(df_list, mode="train", seq_len=sequence_length, num_feat=num_feat, train_split=train_split)
    test_dataset = data_process.pre_train_process_data(df_list, mode="test", seq_len=sequence_length, num_feat=num_feat, train_split=train_split)
    
    new_train_dataset = apply_masking(train_dataset, random_ratio=mask_ratio, param_ratio=mask_ratio, station_ratio=mask_ratio, temporal_ratio=mask_ratio)
    new_test_dataset = apply_masking(test_dataset, random_ratio=mask_ratio, param_ratio=mask_ratio, station_ratio=mask_ratio, temporal_ratio=mask_ratio)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

def fine_tune_load_dataset(dataset_dir, W_dataset_file, M_dataset_file, sequence_length, pre_len, num_feat, M_num_feat, batch_size, train_split):
    W_df, M_df, scaler = data_process.fine_tune_read_dataset(dataset_dir, W_dataset_file, M_dataset_file)
    
    W_train_dataset = data_process.fine_tune_process_data(W_df, mode="train", seq_len=sequence_length, num_feat=num_feat, t=pre_len, train_split=train_split)
    W_test_dataset = data_process.fine_tune_process_data(W_df, mode="test", seq_len=sequence_length, num_feat=num_feat, t=pre_len, train_split=train_split)
    M_train_dataset = data_process.fine_tune_process_data(M_df, mode="train", seq_len=sequence_length, num_feat=M_num_feat, t=pre_len, train_split=train_split)
    M_test_dataset = data_process.fine_tune_process_data(M_df, mode="test", seq_len=sequence_length, num_feat=M_num_feat, t=pre_len, train_split=train_split)
    
    # 创建 DataLoader
    W_train_loader = DataLoader(W_train_dataset, batch_size, shuffle=True)
    W_test_loader = DataLoader(W_test_dataset, batch_size, shuffle=False)
    M_train_loader = DataLoader(M_train_dataset, batch_size, shuffle=True)
    M_test_loader = DataLoader(M_test_dataset, batch_size, shuffle=False)
    
    return W_train_loader, W_test_loader, M_train_loader, M_test_loader, scaler
    

if __name__ == '__main__':
    config = model_config
    
    dataset_files = data_process.get_dataset_files(os.path.join(config.base_dir, config.data['dataset_dir']), '黑龙江')
    config.data['W_data_name'] = '淮河_4.csv'
    config.data['M_data_name'] = 'Meteorology_淮河_4.csv'

    #加载数据集
    # train_loader, test_loader, scaler = pre_train_load_dataset(dataset_dir=os.path.join(config.base_dir, config.data['dataset_dir']),
    #                                                                                                 dataset_files=config.data['WQ_data_name'], sequence_length=config.gsgcn['n_in'],
    #                                                                                                 num_feat=config.gsgcn['input_dim'], batch_size=config.data['batch_size'],
    #                                                                                                train_split=config.data['train_ratio'], mask_ratio=config.data['mask_ratio'])

    W_train_loader, W_test_loader, M_train_loader, M_test_loader, scaler = fine_tune_load_dataset(dataset_dir=os.path.join(config.base_dir, config.data['dataset_dir']), W_dataset_file=config.data['W_data_name'],
                                                               M_dataset_file=config.data['M_data_name'], sequence_length=config.model['n_in'],
                                                               num_feat=config.model['input_dim'], M_num_feat=config.model['feature_dim'],
                                                               batch_size=config.data['batch_size'], train_split=config.data['train_ratio'])
    # 检查加载的数据
    for batch_x, batch_y in M_train_loader:
        batch_x = batch_x.unsqueeze(-1)
        print("训练数据 (新掩码后):", batch_x.shape)  # 应输出形状为 (batch_size, seq_len, num_feat, num_stations)
        break

    for batch_x, batch_y in W_test_loader:
        print("测试数据 (新掩码后):", batch_x.shape)  # 应输出形状为 (batch_size, seq_len, num_feat, num_stations)
        break
