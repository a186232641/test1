import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GNN(nn.Module):
    def __init__(self, enc_in, dec_in, d_model):
        super(GNN, self).__init__()
        hidden_model = 16
        self.conv1 = GCNConv(enc_in, hidden_model)
        self.conv2 = GCNConv(hidden_model, enc_in)

    def forward(self, input_x,edge_index):
        print("Input features size:", input_x.size())  # 打印x的大小
        print("Edge index size:", edge_index.size())  # 打印edge_index的大小
        print("Max index in edge_index:", edge_index.max().item())  # 打印edge_index的最大值
        print("Min index in edge_index:", edge_index.min().item())
        x = self.conv1(input_x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
batch_size = 32
seq_len = 1
station_num = 184
x = torch.randn((batch_size, seq_len, station_num)).to(device)
data = GNN(184,18,18)
data.to(device)
edge_index = pd.read_csv('./data/distance_adj.csv', header=None)
edge_index_temp = sp.coo_matrix(edge_index)
indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
laplace_adj = torch.LongTensor(indices).to(device)  # 我们真正需要的coo形
edge_index = laplace_adj
output = data(x,edge_index)
print(data)
