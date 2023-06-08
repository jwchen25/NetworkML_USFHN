import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, GATConv, GATv2Conv, SAGEConv, TransformerConv, GENConv
from models.conv import GCNConv, GINConv


class GNN(nn.Module):
    """ general GNN model using different GraphConv layers 
    """
    def __init__(self, node_in, edge_in, hidden_dim, out_dim, conv_num_layer=3,
                 gnn_type='gin', mlp_layer=2, dropout=0.0, JK="last", residual=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN, self).__init__()
        self.num_layer = conv_num_layer
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
 
        if conv_num_layer < 2:
            raise ValueError("Number of GraphConv layers must be greater than 1.")
        if mlp_layer < 1:
            raise ValueError("Number of output MLP layers must be >= 1.")

        self.node_l = nn.Linear(node_in, hidden_dim)
        self.edge_l = nn.Linear(edge_in, hidden_dim)

        ### List of Graph convolution layers
        self.convs = torch.nn.ModuleList()

        for layer in range(conv_num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(hidden_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8,
                                  concat=False, edge_dim=hidden_dim))
            elif gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=8,
                                  concat=False, edge_dim=hidden_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'trans':
                self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=4,
                                  concat=False, edge_dim=hidden_dim))
            elif gnn_type == 'deeper':
                self.convs.append(GENConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

        self.mlp = MLP([hidden_dim] * mlp_layer + [out_dim])
        self.act = nn.Sigmoid()  # for output between 0-1

    def reset_parameters(self):
        self.node_l.reset_parameters()
        self.edge_l.reset_parameters()
        self.mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = self.node_l(data.x)
        edge_index = data.edge_index
        edge_attr = self.edge_l(data.edge_attr)

        h_list = [x]
        for layer in range(self.num_layer):
            if self.gnn_type == 'sage':
                h = self.convs[layer](h_list[layer], edge_index)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            node_emb = 0
            for layer in range(self.num_layer + 1):
                node_emb += h_list[layer]
        
        out = self.mlp(node_emb)
        out = self.act(out)
        return out.view(-1)


class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.conv1 = GATConv(input_size, hidden_size, heads=4)
        self.conv2 = GATConv(hidden_size * 4, hidden_size, heads=4)
        self.conv3 = GATConv(hidden_size * 4, hidden_size, heads=6, concat=False)
        self.mlp = MLP([hidden_size, hidden_size, output_size])
        self.elu = nn.ELU()
        self.act = nn.Sigmoid()  # for output between 0-1

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.conv2(x, edge_index)
        x = self.elu(x)
        x = self.conv3(x, edge_index)
        x = self.mlp(x)
        x = self.act(x)
        return x.view(-1)
