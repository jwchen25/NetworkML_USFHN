import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score

from utils import Logger, seed_everything
from dataset import USFHNDataset
from models import GNN


def train(model, data, train_idx, optimizer):
    """ train the model in one epoch
    """
    model.train()
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    out = model(data)[train_idx]
    loss = criterion(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    """ evaluate model on train, valid and test set
    """
    model.eval()
    y_pred = model(data)

    train_res = evaluator(
        y_pred[split_idx['train']],
        data.y[split_idx['train']]).item()

    valid_res = evaluator(
        y_pred[split_idx['valid']],
        data.y[split_idx['valid']]).item()

    test_res = evaluator(
        y_pred[split_idx['test']],
        data.y[split_idx['test']]).item()

    return train_res, valid_res, test_res


def main():
    parser = argparse.ArgumentParser(description='USFHN Ranking')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--subset', type=str, default='chemistry')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--method', type=str, default='trans')
    parser.add_argument('--conv_layers', type=int, default=3)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # in this work, I just use CPU to train the models
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load dataset and create model
    dataset = USFHNDataset(name=args.subset, root='dataset/pyg_data')
    data = dataset[0]
    data = data.to(device)

    model = GNN(node_in=5, edge_in=2, hidden_dim=args.hidden_dim, out_dim=1,
                conv_num_layer=args.conv_layers, gnn_type=args.method,
                mlp_layer=args.mlp_layers, dropout=args.dropout, JK='sum')

    evaluator = r2_score
    logger = Logger(args.runs, args)

    for run in range(args.runs):

        # set global seed for this run
        seed_everything(seed=run)
        # split data for each run
        split_idx = dataset.get_idx_split(seed=run)
        for _idx in split_idx.keys():
            split_idx[_idx].to(device)
        train_idx = split_idx['train']
        # initialize model etc.
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',factor=0.7, patience=5, min_lr=0.00001)

        # training
        for epoch in range(1, 1 + args.epochs):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            scheduler.step(result[1])

            if epoch % args.log_steps == 0:
                train_res, valid_res, test_res = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'lr: {lr:.6f}, '
                      f'Train: {train_res:.4f}, '
                      f'Valid: {valid_res:.4f}, '
                      f'Test: {test_res:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()