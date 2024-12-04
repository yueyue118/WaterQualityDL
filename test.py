import numpy as np
import pandas as pd
import torch
import module_metric

def eval_metrics(output, target):
    # mae
    acc_mae = module_metric.mae_np(output, target)
    # rmse
    acc_rmse = module_metric.rmse_np(output, target)
    # mape
    acc_mape = module_metric.mape_np(output, target)

    n_out = output.shape[1]
    num_nodes = output.shape[2]
    nodes_mae = []
    nodes_rmse = []
    nodes_mape = []
    steps_mae = []
    steps_rmse = []
    steps_mape = []

    # TODO: add step rmse
    for i in range(n_out):
        steps_mae.append(module_metric.mae_np(output[:, i, ...], target[:, i, ...]))
        steps_rmse.append(module_metric.rmse_np(output[:, i, ...], target[:, i, ...]))
        steps_mape.append(module_metric.mape_np(output[:, i, ...], target[:, i, ...]))

    for i in range(num_nodes):
        nodes_mae.append(module_metric.mae_np(output[..., i, :, :], target[..., i, :, :]))
        nodes_rmse.append(module_metric.rmse_np(output[..., i, :, :], target[..., i, :, :]))
        nodes_mape.append(module_metric.mape_np(output[..., i, :, :], target[..., i, :, :]))

    return acc_mae, acc_rmse, acc_mape, nodes_mae, nodes_rmse, nodes_mape, steps_mae, steps_rmse, steps_mape

def predict(model, config, scaler, W_test_loader, M_test_loader):
    model.eval()
    outputs = []
    targets = []
    device = config.device

    with torch.no_grad():
        for (batch_idx, (data, target)), (m_batch_idx, (m_data, m_target)) in zip(enumerate(W_test_loader), enumerate(M_test_loader)):
            data = data.to(device)
            m_data = m_data.to(device)
            output = model(data, m_data)
            output = output.cpu()
            output = output.detach().numpy()
            target = target.unsqueeze(-1).numpy()
            
            output = scaler.inverse_transform(output)
            target = scaler.inverse_transform(target)                     
            outputs.append(np.array(output).reshape(-1, 4).T)
            targets.append(np.array(target).reshape(-1, 4).T)
    outputs = np.hstack(outputs)
    targets = np.hstack(targets)
    return outputs, targets 