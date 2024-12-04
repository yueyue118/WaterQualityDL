import os
import numpy as np
import pandas as pd
import torch
import joblib
import model_config

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# Read all data files
def get_dataset_files(dataset_dir, river):
    dataset_files = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv") and river in filename and 'Meteorology' not in filename:
            dataset_files.append(filename)
    return dataset_files

def pre_train_read_dataset(dataset_dir, dataset_files):
    """
    读取原始数据再合并
    Returns: df_merged
        index    parameter1 parameter2 ... parameter1 parameter2 ... parameter1 parameter2 ...
        datetime
        
    """
    dfs = []
    # Read each dataset and set datetime as index
    for dataset_file in dataset_files:
        df = pd.read_csv(os.path.join(dataset_dir, dataset_file), encoding='utf-8', header=0)
        df = df[['CODMn', 'DO', 'NH4N', 'pH']].to_numpy(dtype=np.float32)
        dfs.append(df)
    # Merge dataframes on datetime index
    df_merged = np.concatenate(dfs, axis=0)
    scaler_data = df_merged.reshape(-1)
    scaler = StandardScaler(mean=scaler_data.mean(), std=scaler_data.std())
    joblib.dump(scaler, os.path.join(dataset_dir, "scaler" + ".pkl"))
    
    for i in range(len(dfs)):
        dfs[i] = scaler.transform(dfs[i])
        
    return dfs, scaler

def fine_tune_read_dataset(dataset_dir, W_dataset_file, M_dataset_file):
    W_df = pd.read_csv(os.path.join(dataset_dir, W_dataset_file), encoding='utf-8', header=0)
    W_df = W_df[['CODMn', 'DO', 'NH4N', 'pH']].to_numpy(dtype=np.float32)
    scaler_data = W_df.reshape(-1)
    scaler = StandardScaler(mean=scaler_data.mean(), std=scaler_data.std())
    joblib.dump(scaler, os.path.join(dataset_dir, "scaler" + ".pkl"))
    W_df = scaler.transform(W_df)
    
    M_df = pd.read_csv(os.path.join(dataset_dir, M_dataset_file), encoding='utf-8', header=0)
    M_df = M_df[['lrad', 'temp', 'pres', 'shum', 'wind', 'srad', 'prec']].to_numpy(dtype=np.float32)
    M_scaler_data = M_df.reshape(-1)
    M_scaler = StandardScaler(mean=M_scaler_data.mean(), std=M_scaler_data.std())
    M_df = M_scaler.transform(M_df)
    return W_df, M_df, scaler

# Split train and test
class pre_train_process_data(torch.utils.data.Dataset):
    def __init__(self, df_list, mode="train", seq_len=6, num_feat=4, train_split=0.8):
        super().__init__()
        self.num_feat = num_feat
        self.num_stations = len(df_list)
        self.seq_len = seq_len
        self.train_split = train_split

        # 存储所有站点的数据
        self.data_list = []

        # 处理每个站点的数据
        for df in df_list:
            # 选择特征列：CODMn, DO, NH4N, pH
            data = df
            dataset_len = len(data)

            # 数据集划分比例：80%训练，20%测试
            train_len = int(self.train_split * dataset_len)

            if mode == "train":
                data = data[:train_len, :]
            else:  # test
                data = data[train_len:, :]

            self.data_list.append(data)

        # 确定所有站点中最短的数据长度，以便统一截取
        min_len = min([len(data) for data in self.data_list])
        
        # 将数据截取到最短长度，并形成最终的格式
        self.data_x = np.zeros((min_len, self.num_feat, self.num_stations), dtype=np.float32)
        for i, data in enumerate(self.data_list):
            self.data_x[:, :, i] = data[:min_len, :]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        # 获取时间序列片段 (seq_len, input_dim, num_stations)
        seq_x = self.data_x[s_begin:s_end, :, :]

        return seq_x, seq_x  # 返回相同的输入和输出

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

class fine_tune_process_data(torch.utils.data.Dataset):
    def __init__(self, df, mode="train", seq_len=6, num_feat=4, t=1, train_split=0.8):
        super().__init__()
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.t = t  # 预测的时间步
        self.train_split = train_split

        # 数据集长度
        dataset_len = len(df)

        # 数据集划分比例：80%训练，20%测试
        train_len = int(self.train_split * dataset_len)

        if mode == "train":
            self.data = df[:train_len, :]
        else:  # test
            self.data = df[train_len:, :]

        # 统一数据长度，确保数据足够长进行序列处理
        self.data_x = np.zeros((len(self.data), self.num_feat), dtype=np.float32)
        self.data_x[:, :] = self.data[:, :]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        pred_end = s_end + self.t

        # 获取时间序列片段 (seq_len, input_dim)
        seq_x = self.data_x[s_begin:s_end, :]

        # 获取接下来的 t 个时间步 (t, input_dim)
        seq_y = self.data_x[s_end:pred_end, :]

        return seq_x, seq_y  # 返回输入序列和对应的预测序列

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.t + 1

if __name__ == '__main__':
    config = model_config
    dataset_files = get_dataset_files(os.path.join(config.base_dir, config.data['dataset_dir']), '黑龙江')
    config.data['WQ_data_name'] = dataset_files
    df_list, scaler = pre_train_read_dataset(dataset_dir=os.path.join(config.base_dir, config.data['dataset_dir']), dataset_files=config.data['WQ_data_name'])

    train_dataset = pre_train_process_data(df_list, mode="train", seq_len=8, num_feat=4)
    test_dataset = pre_train_process_data(df_list, mode="test", seq_len=8, num_feat=4)
    
    print(train_dataset[0][0].shape)
    print(test_dataset[0][0].shape)
    