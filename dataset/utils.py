import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx


def create_graph(node_csv, edge_csv):
    """ create a graph from node and edge csv files

    Args:
        node_csv (str): path of node csv
        edge_csv (str): path of edge csv

    Returns:
        nx.DiGraph: a directed graph
    """
    # load data
    df_node = pd.read_csv(node_csv)
    df_edge = pd.read_csv(edge_csv)

    # check data
    assert len(df_node) == len(df_node['InstitutionId'].unique())
    
    # node attributions
    df_attr = df_node[
        ['InstitutionId', 'NonAttritionEvents',
         'AttritionEvents', 'PrestigeRank']
    ]
    node_attr = df_attr.set_index('InstitutionId').to_dict(orient='index')

    # edge list (source, target, weight) and edge attr
    edge_list = df_edge[
        ['DegreeInstitutionId', 'InstitutionId', 'Total']
    ].values

    # graph
    G = nx.DiGraph()
    G.add_weighted_edges_from(edge_list)
    nx.set_node_attributes(G, node_attr)

    return G


def compute_features(G):
    """ cumpute node and edge features

    Args:
        G (nx.Graph): input graph

    Returns:
        nx.Graph: graph containing computed features
    """
    G = G.copy()
    # degree centrality
    dc = nx.degree_centrality(G)
    nx.set_node_attributes(G, dc, 'degree_centrality')
    # eigenvector centrality
    ec = nx.eigenvector_centrality(G)
    nx.set_node_attributes(G, ec, 'eigenvector_centrality')
    # closeness centrality
    cc = nx.closeness_centrality(G)
    nx.set_node_attributes(G, cc, 'closeness_centrality')
    # betweenness centrality
    bc = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, bc, 'betweenness_centrality')
    # clustering_coefficients
    cl = nx.clustering(G)
    nx.set_node_attributes(G, cl, 'clustering_coefficients')
    # harmonic centrality
    hc = nx.harmonic_centrality(G)
    node_num = len(G.nodes)
    for key in hc.keys():
        hc[key] /= (node_num - 1)
    nx.set_node_attributes(G, hc, 'harmonic_centrality')

    # edge betweenness centrality
    bb = nx.edge_betweenness_centrality(G, normalized=False)
    nx.set_edge_attributes(G, bb, "edge_betweenness_centrality")

    return G


def graph2data(G, normalization=True):
    """ convert graph to PyG data

    Args:
        G (nx.Graph): input graph
        normalization (bool, optional): whether normalize features

    Returns:
        pyg.data: PyG data converted from graph
    """
    G = G.copy()
    rank = nx.get_node_attributes(G, 'PrestigeRank')
    nx.set_node_attributes(G, rank, 'y')
    data = from_networkx(G)
    att_ratio = data.AttritionEvents / (data.AttritionEvents + data.NonAttritionEvents)

    # node features
    node_fea = torch.stack([
        att_ratio,
        data.degree_centrality,
        data.eigenvector_centrality,
        data.closeness_centrality,
        data.harmonic_centrality,
        # data.betweenness_centrality,
        # data.clustering_coefficients,
    ])
    data.x = node_fea.t()
    if normalization:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        data.x = (data.x - mean) / std

    # edge features
    edge_fea = torch.stack([
        data.weight,
        data.edge_betweenness_centrality
    ])
    data.edge_attr = edge_fea.t()
    # if normalization:
    #     max = data.edge_attr.max(dim=0, keepdim=True).values
    #     min = data.edge_attr.min(dim=0, keepdim=True).values
    #     data.edge_attr = (data.edge_attr - min) / (max - min)
    if normalization:
        mean = data.edge_attr.mean(dim=0, keepdim=True)
        std = data.edge_attr.std(dim=0, keepdim=True)
        data.edge_attr = (data.edge_attr - mean) / std

    return data


def feature2csv(G, fpath):
    """ write node features of a graph into a CSV file

    Args:
        G (nx.Graph): input graph
        fpath (str): path of output CSV file

    Returns:
        np.array: node features and ranking
    """
    G = G.copy()
    data = from_networkx(G)
    degree_ratio = data.AttritionEvents / (data.AttritionEvents + data.NonAttritionEvents)

    # node features and ranking
    node_fea = torch.stack([
        degree_ratio,
        data.degree_centrality,
        data.eigenvector_centrality,
        data.harmonic_centrality,
        data.closeness_centrality,
        data.betweenness_centrality,
        data.clustering_coefficients,
        data.PrestigeRank
    ]).t().numpy()

    header = ('degree_ratio,'
              'degree_centrality,'
              'eigenvector_centrality,'
              'harmonic_centrality,'
              'closeness_centrality,'
              'betweenness_centrality,'
              'clustering_coefficients,'
              'PrestigeRank')
    np.savetxt(fpath, node_fea, delimiter=',', header=header, comments='')

    return node_fea


def fit_r2(x, y):
    """ get the R2 value of the linear fit of two numpy arrays

    Args:
        x (np.array): x array
        y (np.array): y array

    Returns:
        float: R2 score
    """
    # Fit a first-degree polynomial (linear fit)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    
    # Calculate predicted values
    y_pred = polynomial(x)
    
    # Calculate the R2 value
    r2 = np.corrcoef(y, y_pred)[0, 1] ** 2
    
    return r2
