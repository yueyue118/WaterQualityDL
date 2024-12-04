import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import copy

class Memory(nn.Module):
    def __init__(self, num_memory, memory_dim):
        super().__init__()
        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.x_proj = nn.Linear(memory_dim, memory_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))
        # x_query: torch.Size([64, 28, 7, 6, memory_dim]) 
        
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM
        # att_weight: torch.Size([64, 28, 7, 6, num_memory])
        
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
        # out: torch.Size([64, 28, 7, 6, memory_dim])
        return out

# 气象影响
class MeteorologyAttention(nn.Module):
    def __init__(self, num_memory, memory_dim, nhead):
        super(MeteorologyAttention, self).__init__()
        self.memory = Memory(num_memory, memory_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=nhead, dim_feedforward=memory_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)       

    def forward(self, water_data, weather_data):
        # water_data & weather_data: (batch_size, seq_len, W_num_feat, num_station) (batch_size, seq_len, M_num_feat, num_station)
        combined_data = torch.cat((water_data.permute(0, 1, 3, 2), weather_data.permute(0, 1, 3, 2)), dim=-1)  # Concatenate along num_station
        # combined_data=[batch_size, seq_len, num_station, W_num_feat+M_num_feat]
        combined_data = combined_data.permute(0, 1, 3, 2)
        # combined_data=[batch_size, seq_len, W_num_feat+M_num_feat, num_station]
        combined_data = combined_data.view(combined_data.size(0), -1, combined_data.size(-1))
        # combined_data=[batch_size, seq_len*(W_num_feat+M_num_feat), num_station]
        modulated_data = self.transformer_encoder(combined_data)
        # modulated_data=[batch_size, seq_len*(W_num_feat+M_num_feat), num_station]        
        modulated_data = modulated_data.view(water_data.size(0), water_data.size(1), water_data.size(2)+weather_data.size(2), water_data.size(3), -1)
        #  modulated_data=[batch_size, seq_len, W_num_feat+M_num_feat, num_station, num_memory]
        ma_output = self.memory(modulated_data)
        #  ma_output=[batch_size, seq_len, W_num_feat+M_num_feat, num_station, num_memory]
        return ma_output

# 知识融合
class CombinedAttentionConcat(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len):
        super(CombinedAttentionConcat, self).__init__()
        self.num_memory = input_dim*2
        self.meteorology_attention = MeteorologyAttention(num_memory=self.num_memory, memory_dim=input_dim, nhead=input_dim)
        # Use a linear layer to reduce the concatenated features back to the original size
        self.mlp = nn.Sequential(nn.Linear(11, 64), nn.ReLU(), nn.Linear(64, 4))
        self.fc = nn.Linear(seq_len, pred_len)
        
    def forward(self, water_data, weather_data):
        water_data = water_data.unsqueeze(-1)
        weather_data = weather_data.unsqueeze(-1)       
        # meteorology_output=[batch_size, seq_len, W_num_feat+M_num_feat, num_station, num_memory]
        meteorology_output = self.meteorology_attention(water_data, weather_data)
        # combined_output=[batch_size, seq_len, W_num_feat+M_num_feat, num_station]
        combined_output = meteorology_output.squeeze(-1)
        # combined_output=[batch_size, seq_len, num_station, W_num_feat+M_num_feat]
        combined_output = combined_output.permute(0, 1, 3, 2)
        # combined_output=[batch_size, seq_len, num_station, W_num_feat]
        combined_output = self.mlp(combined_output)
        # combined_output=[batch_size, num_station, W_num_feat, seq_len]
        combined_output = combined_output.permute(0, 2, 3, 1)
        # combined_output=[batch_size, num_station, W_num_feat, pred_len]
        combined_output = self.fc(combined_output)
        # combined_output=[batch_size, pred_len, W_num_feat, num_station]
        combined_output = combined_output.permute(0, 3, 2, 1)
        return combined_output


if __name__ == '__main__':
    batch_size = 64
    seq_len = 10
    num_feat = 7
    num_station = 1
    
    water_data = torch.rand(batch_size, seq_len, 4, num_station)
    weather_data = torch.rand(batch_size, seq_len, num_feat, num_station)
    combined_attention = CombinedAttentionConcat(num_station)
    output = combined_attention(water_data, weather_data)
    print(output.shape)