import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

from Rating_att_pre import Rating_att_Pre, compute_loss


class MI3Graph(nn.Module):
    def __init__(self, g, mean_rate, in_feats,
                 out_feats,
                 aggregator_type,
                 bias,
                 norm,
                 activation,
                 n_heads,
                 att_boun):
        super(MI3Graph, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU()),
            'trust': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())},
            aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU()),
            'trust': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv3 = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU()),
            'trust': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv4 = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU()),
            'trust': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv1att = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv2att = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv3att = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU())},
            aggregate='max')
        self.conv4att = dglnn.HeteroGraphConv({
            'rated': dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU()),
            'rated-by': dglnn.GraphConv(in_feats, out_feats,  activation=nn.LeakyReLU())},
            aggregate='max')
        self.trust_layer1 = dglnn.GATConv(in_feats, out_feats, num_heads=1, activation=nn.LeakyReLU())
        self.trust_layer2 = dglnn.GraphConv(in_feats, out_feats, activation=nn.LeakyReLU())
        self.Rating_Trust_Pre = Rating_att_Pre(in_feats, out_feats, g, mean_rate)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.dropout3 = torch.nn.Dropout(p=0.1)
        self.dropout4 = torch.nn.Dropout(p=0.1)
        self.l1 = nn.Linear(2 * in_feats, 2 * in_feats)
        self.l2 = nn.Linear(2 * in_feats, in_feats)
        self.attitude_u = nn.Linear(in_feats, in_feats)
        self.attitude_i = nn.Linear(in_feats, in_feats)
        self.att_boun = att_boun

    def forward(self, g, neg, blocks, user_emb, item_emb, trust_emb, alpha, a_emb, helpfulness):
        x = self.conv1(blocks[0], {'uid': user_emb, 'iid': item_emb})
        x = self.conv2(blocks[1], x)
        # x = self.conv3(blocks[2], x)
        # x = self.conv4(blocks[3], x)

        a = self.conv1att(dgl.edge_type_subgraph(blocks[0], ['rated', 'rated-by']), a_emb)
        a = self.conv2att(dgl.edge_type_subgraph(blocks[1], ['rated', 'rated-by']), a)
        # a = self.conv3att(dgl.edge_type_subgraph(blocks[2], ['rated', 'rated-by']), a)
        # a = self.conv4att(dgl.edge_type_subgraph(blocks[3], ['rated', 'rated-by']), a)

        trust_g = dgl.add_self_loop(dgl.edge_type_subgraph(g, ['trust']))
        t = self.trust_layer1(trust_g, trust_emb)
        t = self.dropout1(torch.squeeze(t, 1))
        t = self.trust_layer2(trust_g, t)

        g.dstdata['emb']['uid'] = x['uid']
        g.dstdata['emb']['iid'] = x['iid']
        g.dstdata['trust']['uid'] = t
        g.dstdata['a'] = a
        pos_score, l_att, trust_auc, trust_ap, loss_trust = self.Rating_Trust_Pre(g, neg, x, t, a)  # trust_auc, trust_ap
        rating_loss, mae = compute_loss(g.edges['rated'].data['rating'],
                                        pos_score[('uid', 'rated', 'iid')])
        loss_reg = torch.sum(torch.sum(torch.abs(x['uid'])) + torch.sum(torch.abs(x['iid']))
                             + torch.sum(torch.abs(t)) + torch.sum(torch.abs(a['uid'])) + torch.sum(torch.abs(a['iid'])))
        loss_a_x = torch.sum(torch.sum(torch.abs(x['iid'] - a['iid'])))
        return rating_loss, mae, loss_reg, pos_score[
            ('uid', 'rated', 'iid')], l_att, loss_a_x, trust_auc, trust_ap, loss_trust

    def _eval(self, g, blocks, user_emb, item_emb, trust_emb, a_emb):
        # user_id item_id是测试集的
        x = self.conv1(blocks[0], {'uid': user_emb, 'iid': item_emb})
        x = self.conv2(blocks[1], x)
        # x = self.conv3(blocks[2], x)
        # x = self.conv4(blocks[3], x)

        # a = {'uid': self.attitude_u(x['uid']), 'iid': self.attitude_i(x['iid'])}
        a = self.conv1att(dgl.edge_type_subgraph(blocks[0], ['rated', 'rated-by']), a_emb)
        a = self.conv2att(dgl.edge_type_subgraph(blocks[1], ['rated', 'rated-by']), a)
        # a = self.conv3att(dgl.edge_type_subgraph(blocks[2], ['rated', 'rated-by']), a)
        # a = self.conv4att(dgl.edge_type_subgraph(blocks[3], ['rated', 'rated-by']), a)


        g.dstdata['emb']['uid'] = x['uid']
        g.dstdata['emb']['iid'] = x['iid']
        g.dstdata['a']['uid'] = a['uid']
        g.dstdata['a']['iid'] = a['iid']
        pos_score, l_att, coldr = self.Rating_Trust_Pre._eval(g, x, a, self.att_boun)
        rating_loss, mae = compute_loss(g.edges['rated'].data['rating'],
                                        pos_score[('uid', 'rated', 'iid')])

        return rating_loss, mae, l_att, coldr, g.edges['rated'].data['rating']


