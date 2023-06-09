import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from dataset import USFHNDataset, feature2csv, fit_r2


def plot_fig1():
    """ the function to plot Fig. 1 of the report 
    """
    fig = plt.figure(figsize=(5, 5), layout='constrained')
    subsets = ['chemistry', 'mathematics', 'physics', 'computer_science']

    for i, subset in enumerate(subsets):
        # set sub-figure
        ax = fig.add_subplot(2, 2 , i + 1)
        title = subset.replace('_', ' ').capitalize()
        ax.set_title(title, fontsize=15)
        # load graph (nx.DiGraph)
        dataset = USFHNDataset(name=subset, root='dataset/pyg_data')
        G = dataset.G
        pos = nx.spring_layout(G, seed=0)
        # get top ranking and other institutions (node lists)
        rank = nx.get_node_attributes(G, 'PrestigeRank')
        # get node lists and nodes sizes based on Top 10% ranking
        nodes = np.array(list(rank.keys()))
        ranks = np.array(list(rank.values()))
        ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        ind = np.argsort(-1 * ranks)
        cutoff = int(len(nodes) * 0.10)
        top_node, oth_node = nodes[ind[:cutoff]], nodes[ind[cutoff:]]
        top_rank, oth_rank = ranks[ind[:cutoff]], ranks[ind[cutoff:]]
        top_size = top_rank * 100 + 20
        oth_size = oth_rank * 100 + 20
        # plot graphs via networkx
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, width=1.0,
                               alpha=0.6, edge_color="tab:gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=oth_node, node_color='lightseagreen',
                               node_size=oth_size, alpha=0.6, linewidths=0.1, edgecolors='black')
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=top_node, node_color='crimson',
                               node_size=top_size, alpha=0.8, linewidths=0.8, edgecolors='white')
        # nx.draw_networkx(G, pos, node_size=20, arrows=False, ax=ax, alpha=0.6,
        #                  with_labels=False, edge_color="tab:gray")
        limits = plt.axis("off")
    
    plt.savefig('figures/fig1.pdf')
    plt.savefig('figures/fig1.png')
    plt.show()


def plot_fig2():
    """ the function to plot Fig. 2 of the report 
    """

    # get node features and ranking of 4 subsets
    subsets = ['chemistry', 'mathematics', 'physics', 'computer_science']
    fea_rank = []
    for subset in subsets:
        # load graph (nx.DiGraph)
        dataset = USFHNDataset(name=subset, root='dataset/pyg_data')
        G = dataset.G
        # get node features and ranking and create CSV files 
        fpath = 'dataset/plot_data/' + subset + '_node_features.csv'
        _data = feature2csv(G, fpath)
        fea_rank.append(_data)

    # plot figures
    fig = plt.figure(figsize=(6, 9), layout='constrained')
    col_name = [
        'degree_centrality', 'eigenvector_centrality',
        'harmonic_centrality', 'closeness_centrality',
        'betweenness_centrality', 'clustering_coefficients',
    ]
    col_idx = [1, 2, 3, 4, 5, 6]
    for i, col in enumerate(col_idx):
        ax = fig.add_subplot(3, 2, i + 1)
        title = col_name[i].replace('_', ' ').capitalize()
        ax.set_title(title, fontsize=15)

        # r2_list = np.zeros(len(fea_rank))
        for j, data in enumerate(fea_rank):
            ax.plot(data[:, -1], data[:, col], 'o',
                    label=subsets[j].replace('_', ' '))
        #     r2_list[j] = fit_r2(data[:, -1], data[:, col])
        # print(r2_list.mean())

        ax.set_xlim((0.0, 1.0))
        ax.set_xlabel('Prestige rank', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=14, fontweight='bold')
        plt.tick_params(labelsize=12)
        # ax.legend()

    plt.savefig('figures/fig2.pdf')
    plt.savefig('figures/fig2.png')
    plt.show()


if __name__ == "__main__":

    from matplotlib import rcParams
    rcParams['font.family'] = 'Times New Roman'

    plot_fig2()
