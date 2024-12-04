import pickle as pkl
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
import utils
import model_config
import data_process
import prompt
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

class ModelConfig():

    def __init__(self, args, config, options='', timestamp=True):
        # data
        self.base_dir = config.base_dir
        self.model_name = config.data['model_name']
        self.model_dir = config.data['model_dir']
        self.num_stations = config.data['num_stations']
        self.dataset_dir = config.data['dataset_dir']
        self.model_pkl_filename = config.data['model_pkl_filename']
        self.batch_size = config.data['batch_size']
        self.test_batch_size = config.data['test_batch_size']
        self.data_files = config.data['data_files']
        self.W_data_name = config.data['W_data_name']
        self.M_data_name = config.data['M_data_name']
        self.mask_ratio = config.data['mask_ratio']
        self.train_ratio = config.data['train_ratio']        
        
        # model
        self.n_in = config.model['n_in']
        self.n_out = config.model['n_out']
        self.num_heads = config.model['num_heads']
        self.e_layer = config.model['e_layer']
        self.hidden_size = config.model['hidden_size']
        self.input_dim = config.model['input_dim']
        self.feature_dim = config.model['feature_dim']
        self.output_dim = config.model['output_dim']

        # train
        self.n_gpu = config.train['n_gpu']
        self.epochs = config.train['epochs']
        self.save_dir = config.train['save_dir']
        self.base_lr = config.train['base_lr']
        self.epsilon = config.train['epsilon']
        self.max_grad_norm = config.train["max_grad_norm"]
        self.lr_milestones = config.train['lr_milestones']
        self.lr_decay_ratio = config.train['lr_decay_ratio']

        self.exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # update parameters
        for arg in vars(args):
            if arg == "n_gpu" and getattr(args,arg) is not None:
                setattr(self, arg, getattr(args, arg))
            if hasattr(self, arg) and getattr(args, arg):
                setattr(self, arg, getattr(args, arg))

        self.graph_name = "multichannelgsgcn_%s_nin%d_nout%d_batch%d_epoch%d_time%s" % \
                 (self.W_data_name, self.n_in, self.n_out, self.batch_size, self.epochs, self.exp_time)
        self.device, self.device_ids = utils.prepare_device(self.n_gpu)

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(TimeFeatureEmbedding, self).__init__()
        d_inp = seq_len
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-1)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Transformer_Layer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, num_feat, seq_len, pred_len):
        super(Transformer_Layer, self).__init__()
        self.num_heads = num_heads
        self.e_layer = e_layer
        self.d_model = hidden_size
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.pred_len = pred_len

        # patching and embedding
        self.embedding = TimeFeatureEmbedding(self.d_model, self.seq_len)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.num_feat, attention_dropout=0.1, output_attention=False), self.d_model, self.num_heads),
                    self.d_model, 2048, dropout=0.1, activation='gelu'
                ) for l in range(self.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Prediction Head
        self.head_nf = self.d_model

        self.head = FlattenHead(self.num_feat, self.head_nf, self.pred_len, head_dropout=0.1)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc=[batch_size, seq_len, num_feat]
        x_enc /= stdev
        
        # x_enc=[batch_size, num_feat, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        
        # enc_out=[batch_size, num_feat, d_model]
        enc_out = self.embedding(x_enc)
                
        # Encoder
        # enc_out=[batch_size, num_feat, d_model]
        # attns=[3, ]
        enc_out, attns = self.encoder(enc_out)

        # Decoder
        # dec_out=[batch_size, num_feat, seq_len]
        dec_out = self.head(enc_out)
        
        # dec_out=[batch_size, seq_len, num_feat]  
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        # dec_out=[batch_size, seq_len, num_feat] 
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # dec_out=[batch_size, seq_len, num_feat] 
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out 
    
    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
    
class MultiTransformer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, num_stations, num_feat, seq_len, pred_len):
        super(MultiTransformer, self).__init__()
        self.num_heads = num_heads
        self.e_layer = e_layer
        self.d_model = hidden_size
        self.num_stations = num_stations
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transformers = nn.ModuleList([
            Transformer_Layer(self.num_heads, self.e_layer, self.d_model, self.num_feat, self.seq_len, self.pred_len)
            for _ in range(self.num_stations)])
        
        self.mlp = nn.Linear(self.num_stations, self.num_stations)
        
    def forward(self, x):
        outputs = []
        for i in range(self.num_stations):
            channel_data = x[..., i]
            channel_out = self.transformers[i](channel_data)
            outputs.append(channel_out)
        
        # out=[batch_size, seq_len, num_feat, num_station]
        out = torch.stack(outputs, dim=-1)
        out = self.mlp(out)
        return out

class Prompt_MultiTransformer(nn.Module):
    def __init__(self, num_heads, e_layer, hidden_size, prompt_num, num_stations, num_feat, seq_len, pred_len):
        super(Prompt_MultiTransformer, self).__init__()
        self.num_heads = num_heads
        self.e_layer = e_layer
        self.d_model = hidden_size
        self.num_stations = num_stations
        self.prompt_num = prompt_num
        self.num_feat = num_feat
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transformers = nn.ModuleList([
            Transformer_Layer(self.num_heads, self.e_layer, self.d_model, self.num_feat, self.seq_len, self.pred_len)
            for _ in range(self.num_stations)])
        self.mlp = nn.Linear(self.num_stations, self.prompt_num)
        self.combined_attention = prompt.CombinedAttentionConcat(self.prompt_num, self.seq_len, self.pred_len)
        self.fusion_layer = nn.Sequential(nn.Linear(self.prompt_num*2, 32), nn.ReLU(), nn.Linear(32, self.prompt_num))
        
    def forward(self, x, x_weather):
        outputs = []
        for i in range(self.num_stations):
            channel_data = x
            channel_out = self.transformers[i](channel_data)
            outputs.append(channel_out)
        
        # out=[batch_size, seq_len, num_feat, num_station]
        out = torch.stack(outputs, dim=-1)
        out = self.mlp(out)
        # out_feature=[batch_size, seq_len, num_feat, num_station]
        out_feature = self.combined_attention(x, x_weather)
        # out_combined=[batch_size, seq_len, num_feat, num_station*2]
        out_combined = torch.cat((out, out_feature), dim=-1)
        # output=[batch_size, seq_len, num_feat, num_station]
        output =  self.fusion_layer(out_combined)
        return output


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
    args =parser.parse_known_args()[0]
    config = ModelConfig(args, model_config)

    #加载数据集
    dataset_files = data_process.get_dataset_files(os.path.join(config.base_dir, config.dataset_dir), '黑龙江')
    config.data_files = dataset_files

    #加载数据集
    train_loader, test_loader, scaler = utils.pre_train_load_dataset(dataset_dir=os.path.join(config.base_dir, config.dataset_dir),
                                                                dataset_files=config.data_files, sequence_length=config.n_in,
                                                                num_feat=config.input_dim, batch_size=config.batch_size,
                                                                train_split=config.train_ratio, mask_ratio=config.mask_ratio)
    
    model = MultiTransformer(num_heads=config.num_heads, e_layer=config.e_layer, hidden_size=config.hidden_size,
                             num_stations=len(dataset_files), num_feat=config.input_dim,
                             seq_len=config.n_in, pred_len=config.n_out)
    
    
    for batch_x, batch_y in train_loader:
        print("输入数据:", batch_x.shape)
        y = model(batch_x)
        print("输出数据:", y.shape)
        break
