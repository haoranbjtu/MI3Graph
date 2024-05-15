import random
import pickle
import scipy.io as scio
import scipy.sparse as ssp
import numpy as np
import pandas as pd

import torch
import dgl

from builder import PandasGraphBuilder

random.seed(97)


def build_train_graph(g, train_indices, train_indices_t, etype1, etype_rev1, etype2):
    train_g = dgl.edge_subgraph(g,
        {etype1: train_indices, etype_rev1: train_indices, etype2: train_indices_t},
        relabel_nodes=False)

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix


def build_test_matrix(g, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return test_matrix


if __name__ == '__main__':
    # workdir = '../dataset/ciao_2378_16861'  # change to your workdir
    workdir = '../dataset/epinions_7411_8728'
    if workdir == '../dataset/epinions_7411_8728':
        click_f = np.loadtxt(workdir + '/ratings.txt', delimiter=',', dtype=np.int32)
        rate_line_id = 2
        no_helpfulness = True
    else:
        click_f = np.loadtxt(workdir + '/ratings.txt', dtype=np.int32)
        rate_line_id = 3
        no_helpfulness = False
    trust_f = np.loadtxt(workdir + '/trust.txt', dtype=np.int32)

    click_list = []
    trust_list = []

    user_count = 0
    item_count = 0

    for s in click_f:  # 所有数据
        uid = s[0]
        iid = s[1]
        label = s[rate_line_id]
        if uid > user_count:
            user_count = uid
        if iid > item_count:
            item_count = iid
        if not no_helpfulness:
            helpfulness = s[2]
            click_list.append([uid, iid, label, helpfulness])
        else:
            click_list.append([uid, iid, label])

    pos_list = []  # 只有三元组

    if no_helpfulness:
        for i in range(len(click_list)):
            pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
        random.shuffle(pos_list)
        pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label'])
    else:
        for i in range(len(click_list)):
            pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2], click_list[i][3]))
        random.shuffle(pos_list)
        pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label', 'helpfulness'])

    pos_df['boun3'] = pos_df['label']
    pos_df['boun4'] = pos_df['label']
    pos_df.loc[(pos_df['label'] >= 3), 'boun3'] = 1
    pos_df.loc[(pos_df['label'] < 3), 'boun3'] = 0
    pos_df.loc[(pos_df['label'] >= 4), 'boun4'] = 1
    pos_df.loc[(pos_df['label'] < 4), 'boun4'] = 0
    pos_df.loc[(pos_df['label'] >= 2), 'boun2'] = 1
    pos_df.loc[(pos_df['label'] < 2), 'boun2'] = 0
    pos_df.loc[(pos_df['label'] >= 1), 'boun1'] = 1
    pos_df.loc[(pos_df['label'] < 1), 'boun1'] = 0
    pos_df.loc[(pos_df['label'] >= 5), 'boun5'] = 1
    pos_df.loc[(pos_df['label'] < 5), 'boun5'] = 0
    print(pos_df.mean())
    # print(pos_df)

    train_df = pos_df[:int(0.8 * len(pos_list))]
    test_df = pos_df[int(0.8 * len(pos_list)):len(pos_list)]
    # print(pos_list)
    # train_df = train_df.sort_values(axis=0, ascending=True, by='uid')
    pos_df['train_mask'] = np.ones((len(pos_df),), dtype=bool)
    pos_df['test_mask'] = np.ones((len(pos_df),), dtype=bool)
    pos_df['train_mask'][:int(0.8 * len(pos_list))] = True
    pos_df['train_mask'][int(0.8 * len(pos_list)):len(pos_list)] = False
    pos_df['test_mask'][:int(0.8 * len(pos_list))] = False
    pos_df['test_mask'][int(0.8 * len(pos_list)):len(pos_list)] = True
    train_indices = pos_df['train_mask'].to_numpy().nonzero()[0]
    test_indices = pos_df['test_mask'].to_numpy().nonzero()[0]

    # print(train_index)
    for s in trust_f:
        uid = s[0]
        fid = s[1]
        if uid > user_count or fid > user_count:
            continue
        trust_list.append([uid, fid])

    trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
    trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid').reset_index()
    # trusted_df = trust_df.copy()
    # trusted_df['uid'] = trust_df['fid']  # trust使单向关系
    # trusted_df['fid'] = trust_df['uid']
    # trust_df = pd.concat([trust_df, trusted_df], axis=0).reset_index()

    """
    # 读mat文件
    dataFile1 = '../dataset/epinions_7411_8728/Epinions.mat'
    data1 = scio.loadmat(dataFile1)
    print(data1['rating'])
    data1r = pd.DataFrame(list(data1['rating']), columns=['uid', 'iid', 'label'])
    print(data1r)
    print(data1r['uid'].value_counts())
    """
    """
    #####################################################################################################
    # cold-start dataset
    cold_df = pos_df['uid'].value_counts()
    print(cold_df)
    cold_df = cold_df[cold_df <= 20]
    cold_df = cold_df[cold_df > 2]
    cold_start_users = cold_df.index
    co = pd.DataFrame()
    tr_co = pd.DataFrame()
    tred_co = pd.DataFrame()
    for co_user in cold_start_users:
        co = pd.concat([co, pos_df[pos_df.uid == co_user]], axis=0)
        tr_co = pd.concat([tr_co, trust_df[trust_df.uid == co_user]], axis=0)
        tred_co = pd.concat([tred_co, trust_df[trust_df.fid == co_user]], axis=0)

    tr_co = tr_co.drop(columns=['index'])
    tred_co = tred_co.drop(columns=['index'])
    print(tr_co)
    tr_co = tr_co.merge(tred_co, how='inner')
    print(tr_co)  # 社交关系最后user重新编码也要能对应上
    co['uid'] = co['uid'].astype('category')
    co['iid'] = co['iid'].astype('category')
    d_uid = dict(enumerate(co['uid'].cat.categories))  # 新旧id对应关系
    d_uid = dict(zip(d_uid.values(), d_uid.keys()))  # 交换key和val
    print(d_uid)
    co['uid'] = co['uid'].cat.codes.values + 1
    co['iid'] = co['iid'].cat.codes.values + 1
    co = co.reset_index()
    co = co.drop(columns=['index', 'train_mask', 'test_mask'])
    print(co)
    tr_co['uid'] = [d_uid[x]+1 for x in tr_co['uid']]
    tr_co['fid'] = [d_uid[x]+1 for x in tr_co['fid']]
    print(tr_co)
    mean_rating = np.mean(co['label'])
    print(mean_rating)
    print(max(co['iid']))
    co.to_csv(workdir+'/cold_rating.csv', sep=' ', header=0, index=0)
    tr_co.to_csv(workdir+'/cold_trust.csv', sep=' ', header=0, index=0)

    ##########
    
    user_df = pd.DataFrame([i for i in range(1, max(co['uid']) + 1)], columns=['uid'])
    item_df = pd.DataFrame([i for i in range(1, max(co['iid']) + 1)], columns=['iid'])
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_df, 'uid', 'uid')
    graph_builder.add_entities(item_df, 'iid', 'iid')
    graph_builder.add_binary_relations(co, 'uid', 'iid', 'rated')
    graph_builder.add_binary_relations(co, 'iid', 'uid', 'rated-by')
    graph_builder.add_trust_relations(tr_co, 'uid', 'fid', 'trust')

    g = graph_builder.build()
    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.

    g.edges['rated'].data['rating'] = torch.tensor(co['label'].values, dtype=torch.long)
    g.edges['rated'].data['boun3'] = torch.tensor(co['boun3'].values, dtype=torch.long)
    g.edges['rated'].data['boun4'] = torch.tensor(co['boun4'].values, dtype=torch.long)
    g.edges['rated'].data['boun5'] = torch.tensor(co['boun5'].values, dtype=torch.long)
    g.edges['rated'].data['boun1'] = torch.tensor(co['boun1'].values, dtype=torch.long)
    g.edges['rated'].data['boun2'] = torch.tensor(co['boun2'].values, dtype=torch.long)
    g.edges['rated-by'].data['rating'] = torch.tensor(co['label'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun3'] = torch.tensor(co['boun3'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun4'] = torch.tensor(co['boun4'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun5'] = torch.tensor(co['boun5'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun1'] = torch.tensor(co['boun1'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun2'] = torch.tensor(co['boun2'].values, dtype=torch.long)
    if not no_helpfulness:
        g.edges['rated'].data['helpfulness'] = torch.tensor(co['helpfulness'].values, dtype=torch.long)
        g.edges['rated-by'].data['helpfulness'] = torch.tensor(co['helpfulness'].values, dtype=torch.long)

    indice_t = tr_co['uid'].index

    co['train_mask'] = np.ones((len(co),), dtype=bool)
    co['test_mask'] = np.ones((len(co),), dtype=bool)
    co['train_mask'][:int(0.8 * len(co))] = True
    co['train_mask'][int(0.8 * len(co)):len(co)] = False
    co['test_mask'][:int(0.8 * len(co))] = False
    co['test_mask'][int(0.8 * len(co)):len(co)] = True
    train_indices = co['train_mask'].to_numpy().nonzero()[0]
    test_indices = co['test_mask'].to_numpy().nonzero()[0]

    train_g = build_train_graph(g, train_indices, indice_t, 'rated', 'rated-by', 'trust')
    test_g = build_train_graph(g, test_indices, indice_t, 'rated', 'rated-by', 'trust')

    mean_rating = np.mean(co['label'])
    with open(workdir + '/colddataset.pkl', 'wb') as f:
        pickle.dump(co, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tr_co, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(d_uid, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_rating, f, pickle.HIGHEST_PROTOCOL)
    ##################################################################################################

    """


    user_df = pd.DataFrame([i for i in range(1, user_count+1)], columns=['uid'])
    item_df = pd.DataFrame([i for i in range(1, item_count+1)], columns=['iid'])
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_df, 'uid', 'uid')
    graph_builder.add_entities(item_df, 'iid', 'iid')
    graph_builder.add_binary_relations(pos_df, 'uid', 'iid', 'rated')
    graph_builder.add_binary_relations(pos_df, 'iid', 'uid', 'rated-by')
    graph_builder.add_trust_relations(trust_df, 'uid', 'fid', 'trust')
    # graph_builder.add_trust_relations(trusted_df, 'uid', 'fid', 'trust')

    g = graph_builder.build()
    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.

    g.edges['rated'].data['rating'] = torch.tensor(pos_df['label'].values, dtype=torch.long)
    g.edges['rated'].data['boun3'] = torch.tensor(pos_df['boun3'].values, dtype=torch.long)
    g.edges['rated'].data['boun4'] = torch.tensor(pos_df['boun4'].values, dtype=torch.long)
    g.edges['rated'].data['boun5'] = torch.tensor(pos_df['boun5'].values, dtype=torch.long)
    g.edges['rated'].data['boun1'] = torch.tensor(pos_df['boun1'].values, dtype=torch.long)
    g.edges['rated'].data['boun2'] = torch.tensor(pos_df['boun2'].values, dtype=torch.long)
    g.edges['rated-by'].data['rating'] = torch.tensor(pos_df['label'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun3'] = torch.tensor(pos_df['boun3'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun4'] = torch.tensor(pos_df['boun4'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun5'] = torch.tensor(pos_df['boun5'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun1'] = torch.tensor(pos_df['boun1'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun2'] = torch.tensor(pos_df['boun2'].values, dtype=torch.long)
    if not no_helpfulness:
        g.edges['rated'].data['helpfulness'] = torch.tensor(pos_df['helpfulness'].values, dtype=torch.long)
        g.edges['rated-by'].data['helpfulness'] = torch.tensor(pos_df['helpfulness'].values, dtype=torch.long)

    indice_t = trust_df['uid'].index
    # Build the graph with training interactions only.
    # print(indice_t)
    # print(g.edata)

    train_g = build_train_graph(g, train_indices, indice_t, 'rated', 'rated-by', 'trust')
    test_g = build_train_graph(g, test_indices, indice_t, 'rated', 'rated-by', 'trust')

    # test_matrix = build_test_matrix(g, test_indices, 'uid', 'iid', 'rated')
    # print(train_g)
    # print(test_matrix)
    mean_rating = np.mean(pos_df['label'])
    
    with open(workdir + '/dataset.pkl', 'wb') as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trust_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_rating, f, pickle.HIGHEST_PROTOCOL)








