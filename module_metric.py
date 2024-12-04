import numpy as np

def mse_np(pred, target):
    assert pred.shape == target.shape, "Input and output arrays must have the same shape"
    mse = np.mean((pred - target) ** 2)
    return mse

def mae_np(pred, target):
    assert pred.shape == target.shape, "Input and output arrays must have the same shape"
    mae = np.mean(np.abs(pred - target))
    return mae

def rmse_np(pred, target):
    assert pred.shape == target.shape, "Input and output arrays must have the same shape"
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return rmse

def mape_np(pred, target, epsilon=1e-5):
    assert pred.shape == target.shape, "Input and output arrays must have the same shape"
    target = np.clip(target, epsilon, None)
    mape = np.mean(np.abs((pred - target) / target))
    return mape

def nse_np(pred, target):
    assert pred.shape == target.shape, "Input and output arrays must have the same shape"
    mean_observed = np.mean(target)
    numerator = np.sum((target - pred) ** 2)
    denominator = np.sum((target - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

if __name__ == '__main__':
    batch_size = 32
    seq_len = 10
    num_feat = 4
    num_station = 7

    # 随机生成输入和输出数组
    input_array = np.random.randn(batch_size, seq_len, num_feat, num_station)
    output_array = np.random.randn(batch_size, seq_len, num_feat, num_station)

    # 计算输入和输出之间的MSE
    mse_value = mse_np(input_array, output_array)
    mae_value = mae_np(input_array, output_array)
    rmse_value = rmse_np(input_array, output_array)
    mape_value = mape_np(input_array, output_array)
    print('MSE:', mse_value)
    print('MAE:', mae_value)
    print('RMSE:', rmse_value)
    print('MAPE:', mape_value)