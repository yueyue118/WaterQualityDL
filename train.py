import numpy as np
import torch
import module_metric
import time

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

def fine_tune_eval_metrics(output, target):
    # mae
    acc_mae = module_metric.mae_np(output, target)
    # rmse
    acc_rmse = module_metric.rmse_np(output, target)
    # mape
    acc_mape = module_metric.mape_np(output, target)

    n_out = output.shape[1]
    steps_mae = []
    steps_rmse = []
    steps_mape = []

    # TODO: add step rmse
    for i in range(n_out):
        steps_mae.append(module_metric.mae_np(output[:, i, ...], target[:, i, ...]))
        steps_rmse.append(module_metric.rmse_np(output[:, i, ...], target[:, i, ...]))
        steps_mape.append(module_metric.mape_np(output[:, i, ...], target[:, i, ...]))
    return acc_mae, acc_rmse, acc_mape, steps_mae, steps_rmse, steps_mape

def pre_train(model, loss_func, optimizer, config, train_loader, val_loader, lr_scheduler, len_epoch, val_len_epoch):
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

        for batch_idx, (data, target) in enumerate(train_loader):
            # data = data.permute(0, 1, 2, 4, 3)
            # target = target.permute(0, 1, 2, 4, 3)
            data = data.to(device)
            optimizer.zero_grad()
            # TODO: need edit
            output = model(data)
            output = output.cpu()
            # loss is self-defined, need cpu input
            loss = loss_func(output, target)
            train_iteration = e_i * len_epoch + batch_idx
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _, _, _, _ = eval_metrics(output.detach().cpu().numpy(), target.numpy())
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
        if e_i % 1 == 0:
            val_loss, val_mae, val_rmse, val_mape = val(model, loss_func, optimizer, config,
                                                        valid_data_loader=val_loader, val_len_epoch=val_len_epoch)
            print(f"train_epoch_time: {train_epoch_time}")
            # TODO: make this MSE and make it work for multiple inputs
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

def fine_tuning(model, loss_func, optimizer, config, W_train_loader, M_train_loader, W_val_loader, M_val_loader, lr_scheduler, len_epoch, val_len_epoch):
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
            output = model(data, m_data)
            output = output.cpu()
            # loss is self-defined, need cpu input           
            loss = loss_func(output, target.unsqueeze(-1))
            train_iteration = e_i * len_epoch + batch_idx
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _ = fine_tune_eval_metrics(output.detach().cpu().numpy(), target.unsqueeze(-1).numpy())
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
            val_loss, val_mae, val_rmse, val_mape = test(model, loss_func, optimizer, config,
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

def val(model, loss_func, optimizer, config, valid_data_loader, val_len_epoch):
    model.eval()
    total_val_loss = 0
    #total_val_metrics = np.zeros(len(self.metrics))
    total_acc_rmse = 0
    total_acc_mae = 0
    total_acc_mape = 0
    device = config.device

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_data_loader):
            # data = data.permute(0, 1, 2, 4, 3)
            # target = target.permute(0, 1, 2, 4, 3)
            data = data.to(device)
            output = model(data)
            batch_size = data.shape[0]
            output = output.cpu()
            loss = loss_func(output, target)
            total_val_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _, _, _, _ = eval_metrics(output.detach().numpy(), target.numpy())
            total_acc_mae += acc_mae
            total_acc_rmse += acc_rmse
            total_acc_mape += acc_mape

        val_loss = total_val_loss / val_len_epoch
        val_mae = total_acc_mae / val_len_epoch
        val_rmse = total_acc_rmse / val_len_epoch
        val_mape = total_acc_mape / val_len_epoch

    return val_loss, val_mae, val_rmse, val_mape

def test(model, loss_func, optimizer, config, W_test_loader, M_test_loader, val_len_epoch):
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
            batch_size = data.shape[0]
            output = output.cpu()
            loss = loss_func(output, target.unsqueeze(-1))
            total_val_loss += loss.item()
            acc_mae, acc_rmse, acc_mape, _, _, _ = fine_tune_eval_metrics(output.detach().numpy(), target.unsqueeze(-1).numpy())
            total_acc_mae += acc_mae
            total_acc_rmse += acc_rmse
            total_acc_mape += acc_mape

        val_loss = total_val_loss / val_len_epoch
        val_mae = total_acc_mae / val_len_epoch
        val_rmse = total_acc_rmse / val_len_epoch
        val_mape = total_acc_mape / val_len_epoch

    return val_loss, val_mae, val_rmse, val_mape