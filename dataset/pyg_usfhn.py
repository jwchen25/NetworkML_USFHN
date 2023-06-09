import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset

from dataset.utils import create_graph, compute_features, graph2data


NODE_NUMS = {
    'chemistry': 172,
    'mathematics': 161, 
    'physics': 156, 
    'computer_science': 166,
}


def random_split(idx, frac_train=0.7, frac_valid=0.1, frac_test=0.2, seed=None):
    """ Splits data idx randomly into train/validation/test.

    Args:
        idx (array-like): data idx
        frac_train (float, optional): The fraction of training data.
        frac_valid (float, optional): The fraction of valid data.
        frac_test (float, optional): The fraction of test data.
        seed (int, optional): Random seed to use

    Returns:
        Tuple: A tuple of train, valid and test set indices.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is not None:
        np.random.seed(seed)
    num_datapoints = len(idx)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(idx)
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])


class USFHNDataset(InMemoryDataset):
    """ The US faculty hiring network dataset using PyG
    """

    def __init__(self, name, root, transform=None, pre_transform=None):
        """
        Args:
            name (str): name of the subset
            root (str): root directory to store the dataset folder
            transform, pre_transform (optional): transform/pre-transform graph objects
        """
        assert name in [
            'chemistry', 'mathematics', 'physics', 'computer_science'
        ], 'Invalid subset name!'
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.G = nx.read_gpickle(self.processed_paths[1])

    @property
    def raw_file_names(self):
        node_csv = self.name + '_node.csv'
        edge_csv = self.name + '_edge.csv'
        return [node_csv, edge_csv]

    @property
    def processed_file_names(self):
        fname = self.name + '.pt'
        gname = self.name + '.gpickle'
        return [fname, gname]

    def get_idx_split(self, seed=None, train_ratio=0.7,
                      valid_ratio=0.1, test_ratio=0.2):
        idx = range(NODE_NUMS[self.name])

        train_idx, valid_idx, test_idx = random_split(
            idx, train_ratio, valid_ratio, test_ratio, seed
        )

        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def download(self):
        raise RuntimeError(
            'Please run dataset/preprocessing.py first!'
        )

    def process(self):
        G = create_graph(self.raw_paths[0], self.raw_paths[1])
        G = compute_features(G)
        data = graph2data(G)
        data = data if self.pre_transform is None else self.pre_transform(data)

        # save data
        torch.save(self.collate([data]), self.processed_paths[0])
        nx.write_gpickle(G, self.processed_paths[1])
