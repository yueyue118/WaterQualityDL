import torch
import numpy as np

def random_masking(data, mask_ratio):
    mask = torch.rand(data.shape) >= mask_ratio
    masked_data = data.clone()
    masked_data[~mask] = 0
    return masked_data

def parameter_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            for param_idx in range(num_feat):
                if torch.rand(1).item() < mask_ratio:
                    masked_data[i, :, param_idx, station_idx] = 0
    return masked_data

def station_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for station_idx in range(num_stations):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, :, :, station_idx] = 0
    return masked_data

def temporal_masking(data, mask_ratio):
    masked_data = data.clone()
    batch_size, seq_len, num_feat, num_stations = data.shape
    for i in range(batch_size):
        for time_step in range(seq_len):
            if torch.rand(1).item() < mask_ratio:
                masked_data[i, time_step, :, :] = 0
    return masked_data

if __name__ == '__main__':
    # 示例数据
    data = torch.rand((2, 336, 4, 5))  # (batch_size, seq_len, num_feat, num_stations)

    # 1. 随机掩码
    random_masked_data = random_masking(data, mask_ratio=0.1)
    print("随机掩盖后数据:", random_masked_data)

    # 2. 参数掩码
    parameter_masked_data = parameter_masking(data, mask_ratio=0.2)
    print("参数掩盖后数据:", parameter_masked_data)

    # 3. 站点掩码
    station_masked_data = station_masking(data, mask_ratio=0.3)
    print("站点掩盖后数据:", station_masked_data)

    # 4. 时间步掩码
    temporal_masked_data = temporal_masking(data, mask_ratio=0.4)
    print("时间步掩盖后数据:", temporal_masked_data)
