import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
import logging

from MI3Graph_0605 import MI3Graph

logging.basicConfig(level=logging.DEBUG #设置日志输出格式
                    ,filename="./ciao_res.log" #log日志输出的文件位置和文件名
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )

def init_weight(userNum, itemNum, hide_dim):
    initializer = nn.init.xavier_uniform_
    embedding_dict = nn.ParameterDict({
        'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
        'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
    })
    return embedding_dict


def train(g, test_g, mean_rate, args):
    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    emb_dict = init_weight(g.num_nodes('uid'), g.num_nodes('iid'), args.hidden_dims)
    g.nodes['uid'].data['emb'] = emb_dict['user_emb']
    g.nodes['iid'].data['emb'] = emb_dict['item_emb']

    g.nodes['uid'].data['a'] = emb_dict['user_emb']
    g.nodes['iid'].data['a'] = emb_dict['item_emb']

    # for trust
    emb_dict_t = init_weight(g.num_nodes('uid'), g.num_nodes('iid'), args.hidden_dims)
    g.nodes['uid'].data['trust'] = emb_dict_t['user_emb']

    # Sampler
    train_eid_dict = {
        'rated': g.edge_ids(g.edges(etype='rated')[0], g.edges(etype='rated')[1], etype='rated'),
        'rated-by': g.edge_ids(g.edges(etype='rated-by')[0], g.edges(etype='rated-by')[1], etype='rated-by'),
        'trust': g.edge_ids(g.edges(etype='trust')[0], g.edges(etype='trust')[1], etype='trust')}

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,  # exclude='reverse_id',
        # reverse_eids=reverse_eids,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))
    dataloader = dgl.dataloading.DataLoader(
        g, train_eid_dict, sampler,
        batch_size=40480, shuffle=True, drop_last=False)


    # Model
    model = MI3Graph(g, mean_rate, args.hidden_dims, args.hidden_dims,
                     'gcn', True, None, activation=None, n_heads=1, att_boun='boun3').to(device)

    opt = torch.optim.Adam([{'params': model.parameters()}], lr=0.001, weight_decay=0.025)  # 0.025 colddataset 0.038
    dur1 = []
    early_stopping = EarlyStopping(patience=5)

    for epoch_id in range(1, args.num_epochs + 1):
        count_rmse = 0
        count_mae = 0
        count_loss_trust = 0
        count_trust_auc = 0
        count_trust_ap = 0
        count_att = 0
        n = 0
        t0 = time.time()
        model.train()
        for input_nodes, subgraph, neg, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            subgraph = subgraph.to(device)  # 只是blocks最后一张图的dst_nodes  用于预测 和evaluation
            neg = neg.to(device)
            input_features = blocks[0].srcdata['emb']  # srcdata
            trust_features = blocks[0].srcdata['trust']  # srcdata
            a = blocks[0].srcdata['a']  # srcdata
            helpfulness = {'rated': blocks[0].edata['helpfulness'][('uid', 'rated', 'iid')],
                           'rated-by': blocks[0].edata['helpfulness'][('iid', 'rated-by', 'uid')]}
            rating_loss, mae, loss_reg, pos_score, l_att, l_a_x, trust_auc, trust_ap, loss_trust = \
                model(subgraph, neg, blocks, input_features['uid'], input_features['iid'], trust_features['uid'],
                      0.1, a, helpfulness)  # , trust_auc, trust_ap
            # loss优化
            opt.zero_grad()
            (torch.pow(rating_loss, 2) + 0.025 * (torch.pow(loss_reg, 2) + l_a_x) + torch.pow(loss_trust,
                                                                                                    2) +  torch.pow(
                l_att, 2)).backward()
            # 0.1 * loss_trust  0.8*l_a_x
            opt.step()
            count_rmse += rating_loss.item()
            count_mae += mae.item()
            count_loss_trust += loss_trust.item()
            count_att += l_att.item()
            count_trust_auc += trust_auc.item()
            count_trust_ap += trust_ap.item()
            n += 1
        t1 = time.time()
        dur1.append(t1 - t0)
        print(f'The {epoch_id}-th epoch: \n')
        print(
            f'For Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        print(f'For Trust Relation: loss is {count_loss_trust / n}, '
              f'AUC is {count_trust_auc / n} and Average Precision is {count_trust_ap / n}')
        print(f'training time is {np.mean(dur1)}')
        logging.info(f'The {epoch_id}-th epoch: \n')
        logging.info(f'For Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        logging.info(f'For Trust Relation: loss is {count_loss_trust / n}, '
              f'AUC is {count_trust_auc / n} and Average Precision is {count_trust_ap / n}')
        logging.info(f'training time is {np.mean(dur1)}')
        val_rmse = val(test_g, model, mean_rate, args, g.ndata['emb'], g.ndata['a'])
        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping")
            logging.info("Early stopping")
            torch.save(model.state_dict(), f'../ckpt/{args.dataset}_{epoch_id}_model.pt')
            torch.save({'trust': g.ndata['trust'], 'rating': g.ndata['emb'], 'attitude': g.ndata['a']},
                       f'../ckpt/{args.dataset}_{epoch_id}_emb.pt')
            break
    return


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class MyDataset(Dataset):

    def __init__(self, triple):
        self.triple = triple
        self.len = self.triple.shape[0]

    def __getitem__(self, index):
        return self.triple[index, 0], self.triple[index, 1], self.triple[index, 2].float()

    def __len__(self):
        return self.len


def test(g, model, mean_rate, args):
    # 取model
    """
    test_set = torch.Tensor(np.array(test_matrix)).long()
    test_set = MyDataset(test_set)
    test_loader = DataLoader(dataset=test_set, batch_size=5000, shuffle=True)
    """
    dur1 = []
    dur2 = []
    emb = torch.load(f'../ckpt/ciao_86_emb.pt')
    g.ndata['emb'], g.ndata['a'] = emb['rating'], emb['attitude']  # 报错
    g.ndata['trust'] = emb['trust']
    test_nid = {
        'rated': g.edge_ids(g.edges(etype='rated')[0], g.edges(etype='rated')[1], etype='rated'),
        'rated-by': g.edge_ids(g.edges(etype='rated-by')[0], g.edges(etype='rated-by')[1], etype='rated-by')}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=40480, shuffle=True, drop_last=False)  # 4G内存 40000 batch-size

    device = torch.device(args.device)
    if not model:
        model = MI3Graph(g, mean_rate, args.hidden_dims, args.hidden_dims, 'gcn', True, None,
                         activation=None, n_heads=1, att_boun='boun3').to(device)
        model.load_state_dict(torch.load(f'../ckpt/ciao_86_model.pt'))  # 调用最好的 model
    model.eval()
    with torch.no_grad():
        count_rmse = 0
        count_mae = 0
        count_att = 0
        n = 0
        t0 = time.time()

        for input_nodes, subgraph, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            subgraph = subgraph.to(device)  # 只是blocks最后一张图的dst_nodes  用于预测 和evaluation
            input_features = blocks[0].srcdata['emb']  # srcdata
            trust_emb = blocks[0].srcdata['trust']  # srcdata
            a_emb = blocks[0].srcdata['a']  # srcdata
            rating_loss, mae, l_att, _, _ = model._eval(subgraph, blocks, input_features['uid'], input_features['iid'], trust_emb,
                                                  a_emb)
            count_rmse += rating_loss.item()
            count_mae += mae.item()
            count_att += l_att.item()
            n += 1
        t1 = time.time()
        dur1.append(t1 - t0)
        print(
            f'For Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        print(f'test time is {np.mean(dur1)}')
        logging.info(f'For Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        logging.info(f'test time is {np.mean(dur1)}')
    return


def val(g, model, mean_rate, args, x, a):
    dur1 = []
    g.ndata['emb'], g.ndata['a'] = x, a
    test_nid = {
        'rated': g.edge_ids(g.edges(etype='rated')[0], g.edges(etype='rated')[1], etype='rated'),
        'rated-by': g.edge_ids(g.edges(etype='rated-by')[0], g.edges(etype='rated-by')[1], etype='rated-by')}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=40480, shuffle=True, drop_last=False)

    device = torch.device(args.device)
    model.eval()
    with torch.no_grad():
        count_rmse = 0
        count_mae = 0
        count_att = 0
        n = 0
        t0 = time.time()
        cuids = []
        ciids = []
        cpre = []
        clabel = []
        for input_nodes, subgraph, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            subgraph = subgraph.to(device)  # 只是blocks最后一张图的dst_nodes  用于预测 和evaluation
            input_features = blocks[0].srcdata['emb']  # srcdata
            trust_emb = blocks[0].srcdata['trust']  # srcdata
            a_emb = blocks[0].srcdata['a']  # srcdata
            helpfulness = {'rated': blocks[0].edata['helpfulness'][('uid', 'rated', 'iid')],
                           'rated-by': blocks[0].edata['helpfulness'][('iid', 'rated-by', 'uid')]}
            rating_loss, mae, l_att, coldr, label = model._eval(subgraph, blocks, input_features['uid'], input_features['iid'], trust_emb,
                                                  a_emb)
            count_rmse += rating_loss.item()
            count_mae += mae.item()
            count_att += l_att.item()
            n += 1
            (uids, iids, pre) = coldr
            cuids += uids.cpu().numpy().tolist()
            ciids += iids.cpu().numpy().tolist()
            cpre += pre[('uid', 'rated', 'iid')].cpu().numpy().tolist()
            clabel += label.cpu().numpy().tolist()
        cold = [cuids, ciids, cpre, clabel]
        cold = pd.DataFrame(cold)
        cold = pd.DataFrame(cold.values.T, columns=['uid', 'iid', 'pre', 'label'])
        # cold.to_csv('cold_ciao.txt', index=0, header=0)
        t1 = time.time()
        dur1.append(t1 - t0)
        print(
            f'val Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        print(f'val time is {np.mean(dur1)}')
        logging.info(f'val Rating Prediction: RMSE is {count_rmse / n}, MAE is {count_mae / n}, att_loss is {count_att / n}')
        logging.info(f'val time is {np.mean(dur1)}')
    return count_rmse / n


def main():
    import pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../dataset/ciao_2378_16861/dataset.pkl')
    parser.add_argument('--dataset', type=str, default='ciao')
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=64)  # 维度
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cpu')  # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--batches-per-epoch', type=int, default=20000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()
    with open(args.dataset_path, 'rb') as f:
        train_df = pickle.load(f)
        test_df = pickle.load(f)
        trust_df = pickle.load(f)
        train_g = pickle.load(f)
        test_g = pickle.load(f)
        mean_rating = pickle.load(f)

    train(train_g, test_g, mean_rating, args)
    # test(test_g, None, mean_rating, args)
    return


if __name__ == '__main__':
    main()
