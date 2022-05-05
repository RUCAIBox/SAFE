import sys

import networkx as nx
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

# mp_hop, mp_type = None, None

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

def generate_meta_path_from_adj_data(cpnet_graph_path, cpnet_vocab_path, path, num_processes):
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    total_meta_path = set()
    total_count = 0
    max_mp_len = 0
    meta_path_list_dict = {}
    # mp_count_dict = {}
    for name, value in path["input"].items():
        no_mp_fea = 0
        # path_type = path.split('/')[-1].split('.')[0]
        with open(value, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)  # [data1, data2] data1:{"adj":, "concept":}

        with Pool(num_processes) as p:
            meta_path = list(
                tqdm(p.imap(get_meta_path_for_sample, adj_concept_pairs), total=len(adj_concept_pairs))
            )

        meta_path_list_dict[name] = meta_path

        for mp in meta_path:
            mp_s_list, mp_s_set, mp_s_ml = mp
            max_mp_len = max(max_mp_len, mp_s_ml)
            total_meta_path.update(mp_s_set)
            total_count += len(mp_s_list)
            if len(mp_s_list) == 0:
                no_mp_fea += 1
        print("Total %d no mp feature" % (no_mp_fea))
    distinct_count = len(total_meta_path)
    print("Max length of meta-path is:%d" % (max_mp_len))
    print("Total meta path are:%d, distinct meta path are:%d" % (total_count, distinct_count))

    return meta_path_list_dict, total_meta_path


def get_meta_path_for_sample(sample):
    deepth = int(mp_hop)
    n_relation = len(relation2id)
    q_type = 'Q'
    a_type = "A"
    c_type = 'C'
    meta_path_for_sample = []
    # draw_network(sample)
    G = nx.MultiDiGraph()
    adj, concept, qmask, amask, cid2score = sample.values()
    assert len(concept) > 0
    # adj_ = np.zeros((n_relation, len(concept), len(concept)), dtype=np.uint8)
    adj = adj.toarray().reshape(n_relation, len(concept), len(concept))
    for i in range(len(concept)):
        G.add_node(i)
    triples = np.where(adj > 0)
    for r, s, t in zip(*triples):
        G.add_edge(s, t, rel=r)
    max_len = 0
    qnodes = np.where(qmask == True)[0]
    anodes = np.where(amask == True)[0]
    emask = ~(qmask | amask)
    for q in qnodes:
        for a in anodes:
            for edge_path in nx.all_simple_edge_paths(G, source=q, target=a, cutoff=deepth):
                meta_path = ""
                pre = None
                length = 0
                for edge in edge_path:
                    s, t, key = edge
                    rel = G[s][t][key]['rel']
                    s_type = q_type if qmask[s] else c_type if emask[s] else a_type
                    t_type = q_type if qmask[t] else c_type if emask[t] else a_type
                    meta_path = meta_path + s_type + '-' + str(rel) + '-'
                    length += 1
                    if pre == None:
                        pre = t
                    else:
                        assert pre == s, (edge, pre, s)
                        pre = t
                meta_path += t_type
                if c_type not in meta_path:
                    max_len = max(max_len, length)
                    meta_path_for_sample.append(meta_path)
            for edge_path in nx.all_simple_edge_paths(G, source=a, target=q, cutoff=deepth):
                meta_path = ""
                pre = None
                length = 0
                for edge in edge_path:
                    s, t, key = edge
                    rel = G[s][t][key]['rel']
                    s_type = q_type if qmask[s] else c_type if emask[s] else a_type
                    t_type = q_type if qmask[t] else c_type if emask[t] else a_type
                    meta_path = meta_path + s_type + '-' + str(rel) + '-'
                    length += 1
                    if pre == None:
                        pre = t
                    else:
                        assert pre == s, (edge, pre, s)
                        pre = t
                meta_path += t_type
                if c_type not in meta_path:
                    max_len = max(max_len, length)
                    meta_path_for_sample.append(meta_path)
    if len(meta_path_for_sample) == 0:
        length = 1
        max_len = max(max_len, length)
        no_qc_connect_rel_type = n_relation
        meta_path = "Q-" + str(no_qc_connect_rel_type) + "-A"
        meta_path_for_sample.append(meta_path)

    meta_path_for_sample_set = set(meta_path_for_sample)
    return (meta_path_for_sample, meta_path_for_sample_set, max_len)

if __name__ == "__main__":
    datasets = ['csqa','obqa','copa','phys','socialiqa']
    mp_hop = 2
    subgraph_type = 'only.qa.'
    mp_type = 'q2a'

    cpnet_graph_path = "./data/cpnet/conceptnet.en.pruned.graph"
    cpnet_vocab_path = "./data/cpnet/concept.txt"
    
    for dataset in datasets:
        print("Extract relation path from %s"%(dataset))

        path={'input':{'dev': f"./data/{dataset}/graph/dev.{subgraph_type}.subgraph.adj.pk",
                        'test': f"./data/{dataset}/graph/test.{subgraph_type}.subgraph.adj.pk",
                        'train': f"./data/{dataset}/graph/train.{subgraph_type}.subgraph.adj.pk"},
                'output':{"dev": f"./data/{dataset}/graph/dev.{subgraph_type}.subgraph.adj.metapath.{mp_hop}.{mp_type}.seq.pk",
                        "test": f"./data/{dataset}/graph/test.{subgraph_type}.subgraph.adj.metapath.{mp_hop}.{mp_type}.seq.pk",
                        "train": f"./data/{dataset}/graph/train.{subgraph_type}.subgraph.adj.metapath.{mp_hop}.{mp_type}.seq.pk"}
        }
        print("Input: ", path['input']['dev'])
        print("Output: ", path['output']['dev'])

        meta_path_dict_2, total_meta_path_2 = generate_meta_path_from_adj_data(cpnet_graph_path,cpnet_vocab_path,path,num_processes=60)

        max_num_dist_mp = 0
        for k, v in meta_path_dict_2.items():
            meta_path = v
            dist_meta_path = []
            all_meta_path = []
            for mp in meta_path:
                mp_s_list, mp_s_set, mp_s_ml = mp
                if len(mp_s_list) == 0:
                    print(mp_s_list, mp_s_set)
                all_meta_path.append(len(mp_s_list))
                dist_meta_path.append(len(mp_s_set))
            max_num_dist_mp = max(max_num_dist_mp, np.max(dist_meta_path))
            print("mean dist mp",np.mean(dist_meta_path),"min:",np.min(dist_meta_path),'max:',np.max(dist_meta_path))
            print("mean all mp",np.mean(all_meta_path),"min:",np.min(all_meta_path),'max:',np.max(all_meta_path))
        print("max num of dist mp", max_num_dist_mp)

        mp2id = {}
        for idx, mp in enumerate(total_meta_path_2):
            mp2id[mp] = idx
        print("Total different meta path: ", len(mp2id))

        rel_type_num = len(id2relation) + 1
        if subgraph_type == 'only.qa' and mp_type != 'n2c':
            node_type_dict = {'Q': 0, 'A': 1}
        else:
            node_type_dict = {'Q':0,'A':1,'C':2}
        # node_type_dict = {'Q': 0, 'A': 1, 'C': 2}
        print(node_type_dict)
        NODE_TYPE_NUM = len(node_type_dict)
        REL_TYPE_NUM = rel_type_num
        one_hot_fea_len = NODE_TYPE_NUM + int(mp_hop)*(NODE_TYPE_NUM + REL_TYPE_NUM)
        print("One hot fea len:", one_hot_fea_len)
        mp_count_distr_dict = {}
        total_num_dist_mp = len(mp2id)
        for k, v in meta_path_dict_2.items():
            meta_path = v
            with open(path['input'][k], 'rb') as fin:
                adj_concept_pairs = pickle.load(fin)  # [data1, data2] data1:{"adj":, "concept":}
            assert len(meta_path) == len(adj_concept_pairs)
            adj_concept_pairs_new = []
            mp_count_distr_list = []
            for data_dict, mp_list in tqdm(zip(adj_concept_pairs, meta_path), total=len(adj_concept_pairs)):
                all_mp_for_sam, dist_mp_for_sam, _ = mp_list
                mp_seq = np.zeros((total_num_dist_mp, one_hot_fea_len), dtype=np.int8)
                mp_count = np.zeros((total_num_dist_mp), dtype=np.int8)
                mp_count_distr = np.zeros((total_num_dist_mp), dtype=np.int8)
                dist_mp_for_sam = sorted(dist_mp_for_sam, key=lambda mp: mp2id[mp])
                if len(dist_mp_for_sam) == 0:
                    print("Error")
                for _, mp in enumerate(dist_mp_for_sam):
                    idx = mp2id[mp]
                    components = mp.split('-')
                    start = 0
                    for c in components:
                        if c in node_type_dict:
                            cur = start + node_type_dict[c]
                            mp_seq[idx][cur] = 1
                            start += NODE_TYPE_NUM
                        else:
                            cur = start + int(c)
                            mp_seq[idx][cur] = 1
                            start += REL_TYPE_NUM
                        # print(start)
                    mp_count[idx] = all_mp_for_sam.count(mp)
                    mp_count_distr[idx] = all_mp_for_sam.count(mp)
                data_dict['metapath_array_feature'] = (mp_seq, mp_count)
                adj_concept_pairs_new.append(data_dict)
                # mp_count_distr_list.append(mp_count_distr)
            # mp_count_distr_dict[k] = mp_count_distr_list
            with open(path['output'][k], 'wb') as fout:
                pickle.dump(adj_concept_pairs_new, fout)
            print("Save %s to %s" % (path['input'][k], path['output'][k]))

        del meta_path_dict_2, total_meta_path_2, adj_concept_pairs_new