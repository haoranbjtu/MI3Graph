import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-10


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        dis = (yhat - y + EPS)
        mae = torch.sum(torch.abs(dis), dim=0) / (y.shape[0])
        return torch.sqrt(torch.sum(torch.pow(dis, 2), dim=0) / (y.shape[0])), mae


def compute_loss(rating_true, rating_pred):
    """
    calculate rmse using tensor
    :param rating_true:
    :param rating_pred:
    :return:
    """
    rating_pred = rating_pred  # .squeeze(1)
    criterion = RMSELoss()
    rmse, mae = criterion(rating_true, rating_pred)
    return rmse, mae


class Rating_att_Pre(nn.Module):
    def __init__(self, final_features, in_features, g, mean_rate):
        super().__init__()
        n_user = g.number_of_nodes('uid')
        n_item = g.number_of_nodes('iid')
        self.m_rate = mean_rate
        self.ubias = nn.Parameter(torch.rand(n_user))
        self.ibias = nn.Parameter(torch.rand(n_item))
        self.W = nn.Linear(2 * in_features, 1)
        self.W1 = nn.Linear(2 * in_features, in_features)
        self.W2 = nn.Linear(in_features, in_features)
        self.W3 = nn.Linear(in_features, 1)
        self.W4 = nn.Linear(4 * in_features, 2 * in_features)
        self.W5 = nn.Linear(2 * in_features, in_features)
        self.W6 = nn.Linear(in_features, 1)
        self.bn1 = nn.BatchNorm1d(2 * in_features, momentum=0.5)
        self.item_intent = nn.Parameter(0.01 * torch.rand(n_item, in_features))
        self.user_intent = nn.Parameter(0.01 * torch.rand(n_user, in_features))
        self.dropout = torch.nn.Dropout(p=0.1)
        self.emb_size = in_features
        self.W_att = nn.Linear(2 * in_features, 1)
        self.W_so = nn.Linear(2 * in_features, in_features)
        self.W_so2 = nn.Linear(in_features, 1)
        self.ReLU = torch.nn.ReLU()
        self.sbias = nn.Parameter(torch.rand(n_user))

    def _add_bias(self, edges):
        bias_src = self.ubias[edges.src[dgl.NID]]
        bias_dst = self.ibias[edges.dst[dgl.NID]]
        return {'score': edges.data['score'].squeeze(1) + bias_src + bias_dst}

    def solinkpre(self, edges):
        bias_src = self.sbias[edges.src[dgl.NID]]
        bias_dst = self.sbias[edges.dst[dgl.NID]]
        t = torch.cat([edges.src['t'], edges.dst['t']], 1)
        t = torch.sigmoid(self.W_so2(self.W_so(t)))
        return {'score': t.squeeze(1)}  #  + bias_src + bias_dst}

    def _add_influ(self, edges):
        influ = (torch.mm(self.item_intent[edges.dst[dgl.NID]], edges.src['x'].t())  # 隐式反馈
                 + torch.mm(edges.dst['x'], (self.user_intent[edges.src[dgl.NID]].t()))
                 + 5 * torch.mm(self.item_intent[edges.dst[dgl.NID]], edges.dst['x'].t())
                 + 5 * torch.mm(edges.src['x'], (self.user_intent[edges.src[dgl.NID]].t()))
                 + torch.mm(self.item_intent[edges.dst[dgl.NID]], self.user_intent[edges.src[dgl.NID]].t()))
        return {'score': edges.data['score'] + torch.diag(influ)}

    def _add_influ_mlp_concat(self, edges):
        influ = self.W6(self.W5(self.W4(torch.cat((self.item_intent[edges.dst[dgl.NID]], edges.src['x'],
                                                   edges.dst['x'], self.user_intent[edges.src[dgl.NID]]), 1))))
        return {'score': edges.data['score'] + influ.squeeze(1)}

    def rating_pre(self, edges):  # cat, sum, dot
        data = torch.cat((edges.src['x'], edges.dst['x']), 1)  # cat
        data = self.bn1(data)
        return {'score': (self.W3(self.W2(self.W1(data)))).squeeze(1)}
        # return {'score': self.W((data))}  # torch.sigmoid

    def cascade(self, edges):
        cas = torch.diag((1 - torch.exp(EPS + edges.dst['cascade'])) @ edges.src['x'].t())
        att = torch.where(edges.data['att'] >= 0.5, 1.0, -1.0)  # 加正态度 负态度减影响力
        return {'score': cas * att}

    def linear_s(self, edges):
        ls = edges.dst['a_sim']
        att = torch.where(edges.data['att'] >= 0.5, 0, -0)  # 加正态度 负态度减影响力
        return {'score': ls * att}

    def linear(self, edges):
        ls = torch.sigmoid(torch.diag(edges.src['l_neigh'] @ edges.dst['x'].t()))
        att = torch.where(edges.data['att'] >= 0.5, 2.5, -0.5)
        return {'score': ls * att}  # edges.data['score'] +

    def attitude(self, edges):
        bias_src = self.ubias[edges.src[dgl.NID]]
        bias_dst = self.ibias[edges.dst[dgl.NID]]
        x = torch.cat([edges.src['a'], edges.dst['a']], 1)
        att = torch.sigmoid(self.W_att(x))
        return {'att': att.squeeze(1)}  # + bias_src + bias_dst}

    def _eval(self, g, x, a, att_boun):
        g = g.edge_type_subgraph(['rated', 'rated-by'])
        with g.local_scope():
            g.ndata['a'] = a
            g.ndata['x'] = x
            g.ndata['l'] = {'uid': self.user_intent[g.ndata['_ID']['uid']],
                            'iid': self.item_intent[g.ndata['_ID']['iid']]}

            def simple_ls(edges):
                return {'sim_ls': torch.sigmoid(torch.diag(edges.src['l'] @
                                                           edges.dst['x'].t()))}
            g['rated'].update_all(simple_ls, fn.max('sim_ls', 'a_sim'),
                                  etype='rated')  # 试l还是x，试sum还是mean

            def message_func(edges):
                return {'m1': torch.log(
                    EPS + 1 - torch.sigmoid(EPS + torch.diag(edges.src['x'] @ edges.dst['l'].t()))
                    .squeeze(0).repeat(self.emb_size, 1).t() * edges.dst['l'])}
            g.apply_edges(self.attitude, etype='rated')
            g.apply_edges(self.linear_s, etype='rated')
            g.apply_edges(self._add_influ_mlp_concat, etype='rated')  # _mlp_concat
            pos_score = g.edata['score']  # 用两端节点emb生成边的emb
            att = g.edata['att']
            att_label = g.edata[att_boun][('uid', 'rated', 'iid')]  # 选择按3分还是4分分正负态度
            l_att, _ = compute_loss(att_label, att[('uid', 'rated', 'iid')])
            uids, iids = g.edges(etype='rated')

            return pos_score, l_att, (uids, iids, pos_score)
            # 取user_id item_id 的 kqrt emb 出来计算

    def forward(self, subgraph, neg_graph, x, t, a):
        # x是rating_emb, 由user和item两种，t是trust_emb，只有user一种
        subgraph_r = subgraph.edge_type_subgraph(['rated', 'rated-by'])
        with subgraph_r.local_scope():
            subgraph_r.ndata['a'] = a
            subgraph_r.ndata['x'] = x  # 是1 batch的  也是子图的emb
            subgraph_r.ndata['l'] = {'uid': self.user_intent[subgraph.ndata['_ID']['uid']],
                                     'iid': self.item_intent[subgraph.ndata['_ID']['iid']]}

            def simple_ls(edges):
                return {'sim_ls': torch.sigmoid(torch.diag(edges.src['l'] @
                                                           edges.dst['x'].t()))}

            subgraph_r['rated'].update_all(simple_ls, fn.max('sim_ls', 'a_sim'),
                                           etype='rated')  # 试l还是x，试sum还是mean

            def message_func(edges):
                return {'m1': torch.log(
                    EPS + 1 - torch.sigmoid(EPS + torch.diag(edges.src['x'] @ edges.dst['l'].t()))
                    .squeeze(0).repeat(self.emb_size, 1).t() * edges.dst['l'])}

            subgraph_r.apply_edges(self.attitude, etype='rated')
            subgraph_r.apply_edges(self.linear_s, etype='rated')
            subgraph_r.apply_edges(self._add_influ_mlp_concat, etype='rated')  # _mlp_concat
            pos_score = subgraph_r.edata['score']  # 用两端节点emb生成边的emb
            att = subgraph_r.edata['att']
            att_label = subgraph_r.edata['boun3'][('uid', 'rated', 'iid')]  # 选择按3分还是4分分正负态度
            l_att, _ = compute_loss(att_label, att[('uid', 'rated', 'iid')])

        subgraph_t = subgraph.edge_type_subgraph(['trust'])
        with subgraph_t.local_scope():
            subgraph_t.ndata['t'] = t
            subgraph_t.apply_edges(self.solinkpre, etype='trust')  # 线性层 dgl.function.u_dot_v('t', 't', 'score')
            pos_score_t = subgraph_t.edata['score']  # 用两端节点emb生成边的emb

        neg_graph = neg_graph.edge_type_subgraph(['trust'])
        with neg_graph.local_scope():
            neg_graph.ndata['t'] = t
            neg_graph.apply_edges(self.solinkpre, etype='trust')
            neg_score = neg_graph.edata['score']
        score_t = torch.cat([pos_score_t,
                             neg_score])
        label = torch.cat([torch.ones_like(pos_score_t),
                           torch.zeros_like(neg_score)]).long()
        loss_trust, _ = compute_loss(label, score_t)

        y, pred = label.detach().cpu().numpy(), score_t.detach().cpu().numpy()

        return pos_score, l_att, roc_auc_score(y, pred), average_precision_score(y, pred), loss_trust
