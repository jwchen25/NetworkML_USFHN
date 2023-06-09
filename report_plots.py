import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from dataset import USFHNDataset


def plot_fig1():
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


if __name__ == "__main__":
    plot_fig1()
