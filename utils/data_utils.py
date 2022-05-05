import pickle
import os
import numpy as np
import torch
import copy
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AutoTokenizer, RobertaTokenizerFast)
try:
    from transformers import AlbertTokenizer
except:
    pass

import json
from tqdm import tqdm
from multiprocessing import Pool

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']
SPECIAL_TOKENS = ['[PROMPT]','[ANSWER]','[ENDING]']
tokenizer_global = None
num_choices_global = None
max_seq_length_global = None


class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]


            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

class MultiGPUSparseAdjDataBatchGeneratorKBQA(object):
    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]


            # edge_index_all, edge_type_all = self.adj_data
            # #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            # #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            # edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            # edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

class BatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[], adj_data=[], prompt_data=[]):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists
        self.prompt_data = prompt_data
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            # batch_indexes = self._to_device(batch_indexes)
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device(x[batch_indexes]) for x in self.lists]
            prompt_vecs, prompt_counts = self.prompt_data
            prompt_vec_tensors = self._to_device(prompt_vecs[batch_indexes])
            prompt_count_tensors = self._to_device(prompt_counts[batch_indexes])
            batch_prompt = [self._to_device(x[batch_indexes]) for x in self.prompt_data]
            edge_index_all, edge_type_all = self.adj_data
            # edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            # edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes])
            edge_type = self._to_device([edge_type_all[i] for i in batch_indexes])
            yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists, edge_index, edge_type, prompt_vec_tensors, prompt_count_tensors])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)

def load_sparse_adj_data_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    if args.inverse_relation:
        if args.drop_edge is not None:
            ration = args.drop_edge_ration
            drop_type = args.drop_edge
            suffix = '.drop_'+str(ration)+'_'+drop_type
            # suffix = '.drop_'+str(ration)+'_'+drop_type+'_2_0'
            cache_path = adj_pk_path + suffix + '.loaded_cache' if not args.debug else adj_pk_path + suffix + '.loaded_debug_cache'
        else:
            cache_path = adj_pk_path + '.loaded_cache' if not args.debug else adj_pk_path + '.loaded_debug_cache'
    else:
        cache_path = adj_pk_path +'.wo_inverse_rel.loaded_cache' if not args.debug else adj_pk_path + '.wo_inverse_rel.loaded_debug_cache'

    use_cache = True

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)
        adj_concept_pairs = adj_concept_pairs[:32*num_choice] if args.debug else adj_concept_pairs
        n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for exp_idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            assert len(concepts) == len(set(concepts))
            qam = qm | am #or
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[exp_idx] = len(concepts)
            adj_lengths[exp_idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[exp_idx, 1:num_concept] = torch.tensor(concepts +1)  #To accommodate contextnode, original concept_ids incremented by 1
            concept_ids[exp_idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[exp_idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[exp_idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[exp_idx, 0] = 3 #contextnode
            node_type_ids[exp_idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[exp_idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node # i for relation idx, j for row of each node, k for col of each node

            if args.inverse_relation:
                if exp_idx == 0:
                    print("Using inverse relation")
                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate
                half_n_rel += 2 #should be 19 now
                if args.drop_edge in ['context_node', 'subgraph_node']:
                    if exp_idx == 0:
                        print("Drop edge of %s"%(args.drop_edge))
                    if len(extra_i) > 0:
                        extra_i, extra_j, extra_k = torch.tensor(extra_i), torch.tensor(extra_j), torch.tensor(extra_k)
                        mask_extra = extra_k < max_node_num
                        extra_i, extra_j, extra_k = extra_i[mask_extra], extra_j[mask_extra], extra_k[mask_extra]
                        assert len(extra_i) == len(extra_j) == len(extra_k)
                        extra_i_reverse, extra_j_reverse, extra_k_reverse = extra_i + half_n_rel, extra_k, extra_j
                    mask = (j < max_node_num) & (k < max_node_num)
                    i, j, k = i[mask], j[mask], k[mask]
                    i_revers, j_reverse, k_reverse = i + half_n_rel, k, j
                    if args.drop_edge == 'context_node':  # drop edge between context nodes and subgraph nodes
                        ori_i = torch.cat((extra_i, extra_i_reverse), 0)
                        ori_i_len = ori_i.shape[0]
                        if ori_i_len > 0:
                            max_idx = extra_i.shape[0] * 2
                            size = int(max_idx * (1.0 - args.drop_edge_ration))
                            idx = torch.randint(low=0, high=max_idx, size=(size,))
                            idx_1 = idx[idx < extra_i.shape[0]]
                            idx_2 = idx[idx >= extra_i.shape[0]] - extra_i.shape[0]
                            extra_i, extra_j, extra_k = extra_i[idx_1], extra_j[idx_1], extra_k[idx_1]
                            extra_i_reverse, extra_j_reverse, extra_k_reverse = extra_i_reverse[idx_2], extra_j_reverse[idx_2], \
                                                                                extra_k_reverse[idx_2]
                            if exp_idx == 0:
                                new_i = torch.cat((extra_i, extra_i_reverse), 0)
                                new_i_len = new_i.shape[0]
                                ration_tmp = float(ori_i_len - new_i_len) / ori_i_len * 100
                                print("Drop {:.2f}% edge".format(ration_tmp))
                        # else:
                            # print("Original context node is zero.")
                    elif args.drop_edge == 'subgraph_node':
                        ori_i = torch.cat((i, i_revers), 0)
                        ori_i_len = ori_i.shape[0]
                        if ori_i_len > 0:
                            max_idx = i.shape[0] * 2
                            size = int(max_idx * (1.0 - args.drop_edge_ration))
                            idx = torch.randint(low=0, high=max_idx, size=(size,))
                            idx_1 = idx[idx < i.shape[0]]
                            idx_2 = idx[idx >= i.shape[0]] - i.shape[0]
                            i, j, k = i[idx_1], j[idx_1], k[idx_1]
                            i_revers, j_reverse, k_reverse = i_revers[idx_2], j_reverse[idx_2], k_reverse[idx_2]
                            if exp_idx == 0:
                                new_i = torch.cat((i, i_revers), 0)
                                new_i_len = new_i.shape[0]
                                ration_tmp = float(ori_i_len - new_i_len) / ori_i_len * 100
                                print("Drop {:.2f}% edge".format(ration_tmp))
                        # else:
                            # print("Original subgraph node is zero.")

                    i, j, k = torch.cat((i, extra_i, i_revers, extra_i_reverse), 0), torch.cat((j, extra_j, j_reverse, extra_j_reverse), 0), \
                              torch.cat((k, extra_k, k_reverse, extra_k_reverse), 0)  # cat all data
                    # if exp_idx == 0:
                    #     print(i)
                    #     print(j)
                    #     print(k)
                    edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                    edge_type.append(i) #each entry is [E, ]
                elif args.drop_edge in ['common_replace']:
                    if exp_idx == 0:
                        print("Drop edge of %s"%(args.drop_edge))
                    mask = (j < max_node_num) & (k < max_node_num)
                    i, j, k = i[mask], j[mask], k[mask]
                    i_reverse, j_reverse, k_reverse = i + half_n_rel, k, j
                    common_rel_idx = 15 + 2
                    # common_rel_idx = 0 + 2

                    ori_i = torch.cat((i, i_reverse), 0)
                    ori_i_len = ori_i.shape[0]
                    if ori_i_len > 0:
                        max_idx = i.shape[0] * 2
                        size = int(max_idx * args.drop_edge_ration)
                        idx = torch.randint(low=0, high=max_idx, size=(size,))
                        idx_1 = idx[idx < i.shape[0]]
                        idx_2 = idx[idx >= i.shape[0]] - i.shape[0]
                        if exp_idx == 0:
                            print(i[idx_1])
                        i[idx_1] = common_rel_idx
                        if exp_idx == 0:
                            print(i[idx_1])
                            print(i_reverse[idx_2])
                        i_reverse[idx_2] = common_rel_idx
                        if exp_idx == 0:
                            print(i_reverse[idx_2])

                    if len(extra_i) > 0:
                        extra_i, extra_j, extra_k = torch.tensor(extra_i), torch.tensor(extra_j), torch.tensor(extra_k)
                        mask_extra = extra_k < max_node_num
                        extra_i, extra_j, extra_k = extra_i[mask_extra], extra_j[mask_extra], extra_k[mask_extra]
                        extra_i_reverse, extra_j_reverse, extra_k_reverse = extra_i + half_n_rel, extra_k, extra_j
                    ########################

                    i, j, k = torch.cat((i, extra_i, i_reverse, extra_i_reverse), 0), \
                              torch.cat((j, extra_j, j_reverse, extra_j_reverse), 0), \
                              torch.cat((k, extra_k, k_reverse, extra_k_reverse), 0)  # add inverse relations
                    edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                    edge_type.append(i) #each entry is [E, ]
                elif args.drop_edge in ['edge_shuffle']:
                    if exp_idx == 0:
                        print("Shuffle edge of %s"%(args.drop_edge))
                    mask = (j < max_node_num) & (k < max_node_num)
                    i, j, k = i[mask], j[mask], k[mask]
                    i_reverse, j_reverse, k_reverse = i + half_n_rel, k, j
                    # common_rel_idx = 15 + 2
                    # common_rel_idx = 0 + 2

                    ori_i = torch.cat((i, i_reverse), 0)
                    ori_i_len = ori_i.shape[0]
                    if ori_i_len > 0:
                        candidate_rel = np.arange(2 * half_n_rel)
                        candidate_rel = np.concatenate((candidate_rel[2:19], candidate_rel[21:]))

                        max_idx = i.shape[0] * 2
                        size = int(max_idx * args.drop_edge_ration)
                        idx = torch.randint(low=0, high=max_idx, size=(size,))
                        idx_1 = idx[idx < i.shape[0]]
                        idx_2 = idx[idx >= i.shape[0]] - i.shape[0]

                        candidate_rel = np.random.choice(candidate_rel,size=size)
                        candidate_rel = torch.tensor(candidate_rel)
                        assert idx.shape[0] == candidate_rel.shape[0]

                        if exp_idx == 0:
                            print(i[idx_1])
                        i[idx_1] = candidate_rel[0:len(idx_1)]
                        if exp_idx == 0:
                            print(i[idx_1])

                        if exp_idx == 0:
                            print(i_reverse[idx_2])
                        i_reverse[idx_2] = candidate_rel[len(idx_1):]
                        if exp_idx == 0:
                            print(i_reverse[idx_2])

                    if len(extra_i) > 0:
                        extra_i, extra_j, extra_k = torch.tensor(extra_i), torch.tensor(extra_j), torch.tensor(extra_k)
                        mask_extra = extra_k < max_node_num
                        extra_i, extra_j, extra_k = extra_i[mask_extra], extra_j[mask_extra], extra_k[mask_extra]
                        extra_i_reverse, extra_j_reverse, extra_k_reverse = extra_i + half_n_rel, extra_k, extra_j
                    ########################

                    i, j, k = torch.cat((i, extra_i, i_reverse, extra_i_reverse), 0), \
                              torch.cat((j, extra_j, j_reverse, extra_j_reverse), 0), \
                              torch.cat((k, extra_k, k_reverse, extra_k_reverse), 0)  # add inverse relations
                    edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
                    edge_type.append(i)  # each entry is [E, ]
                else:
                    if exp_idx == 0:
                        print("No drop edge")
                    if len(extra_i) > 0:
                        i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                        j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                        k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                    ########################

                    mask = (j < max_node_num) & (k < max_node_num)
                    i, j, k = i[mask], j[mask], k[mask]
                    i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                    edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                    edge_type.append(i) #each entry is [E, ]
            else:
                if exp_idx == 0:
                    print("Don't using inverse relation")
                # Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0)  # rel from contextnode to question concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1)  # rel from contextnode to answer concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # answer concept coordinate

                half_n_rel += 2  # should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
                edge_type.append(i)  # each entry is [E, ]

        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type) #, half_n_rel * 2 + 1

def load_sparse_adj_data_and_metapath_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    if args.inverse_relation:
        cache_path = adj_pk_path +'.loaded_cache' if not args.debug else adj_pk_path + '.loaded_debug_cache'
    else:
        cache_path = adj_pk_path +'.wo_inverse_rel.loaded_cache' if not args.debug else adj_pk_path + '.wo_inverse_rel.loaded_debug_cache'

    use_cache = True

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, metapath_feature = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)
        adj_concept_pairs = adj_concept_pairs[:32*num_choice] if args.debug else adj_concept_pairs
        n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
        metapath_feature = []

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score, metapath_array = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score'], _data['metapath_array_feature']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            metapath_feature.append(torch.tensor(metapath_array,dtype=torch.float))
            assert len(concepts) == len(set(concepts))
            qam = qm | am #or
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts +1)  #To accommodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[idx, 0] = 3 #contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            if args.inverse_relation:
                if idx == 0:
                    print("Using inverse relation")
                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate

                half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]
            else:
                if idx == 0:
                    print("Don't using inverse relation")
                # Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0)  # rel from contextnode to question concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1)  # rel from contextnode to answer concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # answer concept coordinate

                half_n_rel += 2  # should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
                edge_type.append(i)  # each entry is [E, ]

        metapath_feature = torch.stack(metapath_feature, dim=0) # (n_samples, mp_count)
        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, metapath_feature], f)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature)]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    mp_fea_size = metapath_feature.shape[-1]
    return concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, (edge_index, edge_type), mp_fea_size #, half_n_rel * 2 + 1

def load_sparse_adj_data_and_metapathonehot_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    if not args.debug:
        if args.inverse_relation:
            cache_path = adj_pk_path +'.loaded_cache'
        else:
            cache_path = adj_pk_path +'.wo_inverse_rel.loaded_cache'
        use_cache = True
    else:
        use_cache = False
        cache_path = None

    if use_cache and os.path.exists(cache_path):
        use_cache = True
    else:
        use_cache = False

    if use_cache:
        print(cache_path)
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, metapath_feature, metapath_feature_count = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)
        adj_concept_pairs = adj_concept_pairs[:32*num_choice] if args.debug else adj_concept_pairs
        n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
        metapath_feature = []
        metapath_feature_count = []

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score, metapath_array = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score'], _data['metapath_array_feature']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            mp_one_hot_vec, mp_count = metapath_array
            metapath_feature.append(torch.tensor(mp_one_hot_vec, dtype=torch.float))
            metapath_feature_count.append(torch.tensor(mp_count, dtype=torch.float))
            assert len(concepts) == len(set(concepts))
            qam = qm | am #or
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts +1)  #To accommodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[idx, 0] = 3 #contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            if args.inverse_relation:
                if idx == 0:
                    print("Using inverse relation")
                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate

                half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]
            else:
                if idx == 0:
                    print("Don't using inverse relation")
                # Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0)  # rel from contextnode to question concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1)  # rel from contextnode to answer concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # answer concept coordinate

                half_n_rel += 2  # should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
                edge_type.append(i)  # each entry is [E, ]

        metapath_feature = torch.stack(metapath_feature, dim=0) # (n_samples, mp_count)
        metapath_feature_count = torch.stack(metapath_feature_count, dim=0)
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, metapath_feature, metapath_feature_count], f, protocol=4)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count)]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    mp_fea_size = metapath_feature.shape
    return concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count, (edge_index, edge_type), mp_fea_size #, half_n_rel * 2 + 1

def load_sparse_adj_data_and_metapathonehot_for_kbqa_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    print("Loading adj file")
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)
    adj_concept_pairs = adj_concept_pairs[:32*num_choice] if args.debug else adj_concept_pairs
    n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
    # adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    # concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
    # node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
    # node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
    metapath_feature = []
    metapath_feature_count = []

    # adj_lengths_ori = adj_lengths.clone()
    # edge_index = []  # list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    # edge_type = []  # list of size (n_questions, n_choices), where each entry is tensor[E, ]

    for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
        metapath_array = _data['metapath_array_feature']
        mp_one_hot_vec, mp_count = metapath_array
        metapath_feature.append(torch.tensor(mp_one_hot_vec, dtype=torch.float))
        metapath_feature_count.append(torch.tensor(mp_count, dtype=torch.float))

        # edge_index.append(torch.zeros((2, 100)))  # each entry is [2, E]
        # edge_type.append(torch.zeros((100,)))  # each entry is [E, ]

    metapath_feature = torch.stack(metapath_feature, dim=0) # (n_samples, mp_count)
    metapath_feature_count = torch.stack(metapath_feature_count, dim=0)
    print("meta path feature shape: ", metapath_feature.shape)

    # edge_index = list(map(list, zip(*(iter(
    #     edge_index),) * num_choice)))  # list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    # edge_type = list(map(list, zip(*(iter(
    #     edge_type),) * num_choice)))  # list of size (n_questions, n_choices), where each entry is tensor[E, ]

    # concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count)]
    metapath_feature, metapath_feature_count = [x.view(-1, num_choice, *x.size()[1:]) for x in (metapath_feature, metapath_feature_count)]
    print("meta path feature shape: ", metapath_feature.shape)
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    mp_fea_size = metapath_feature.shape
    # return concept_ids, node_type_ids, node_scores, adj_lengths, metapath_feature, metapath_feature_count, (edge_index, edge_type), mp_fea_size #, half_n_rel * 2 + 1
    return metapath_feature, metapath_feature_count, mp_fea_size #, half_n_rel * 2 + 1

def load_sparse_adj_data_with_contextnode_with_triple_prompt(adj_pk_path, max_node_num, num_choice, args):
    if args.inverse_relation:
        cache_path = adj_pk_path +'.loaded_cache' if not args.debug else adj_pk_path + '.loaded_debug_cache'
    else:
        cache_path = adj_pk_path +'.wo_inverse_rel.loaded_cache' if not args.debug else adj_pk_path + '.wo_inverse_rel.loaded_debug_cache'

    use_cache = True

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, \
                triple_prompt_vecs, triple_prompt_count = pickle.load(f)
    else:
        soft_prompt_dim = args.soft_prompt_dim
        max_prompt_num = args.max_prompt_num
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)
        adj_concept_pairs = adj_concept_pairs[:32*num_choice] if args.debug else adj_concept_pairs
        n_samples = len(adj_concept_pairs) #this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) #default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
        triple_prompt_vecs = torch.zeros((n_samples, max_prompt_num, soft_prompt_dim), dtype=torch.float)
        triple_prompt_count = torch.zeros((n_samples,), dtype=torch.long)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score, tri_cls_vec = \
                _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score'], _data['triple_cls_vec']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            #tri_cls_vec: torch.array(num_triple, 1024)
            assert len(concepts) == len(set(concepts))
            qam = qm | am #or
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts +1)  #To accommodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[idx, 0] = 3 #contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            if args.inverse_relation:
                if idx==0:
                    print("Using inverse relation")
                #Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0) #rel from contextnode to question concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1) #rel from contextnode to answer concept
                        extra_j.append(0) #contextnode coordinate
                        extra_k.append(_new_coord) #answer concept coordinate

                half_n_rel += 2 #should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
                edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
                edge_type.append(i) #each entry is [E, ]
            else:
                if idx == 0:
                    print("Don't using inverse relation")
                # Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0)  # rel from contextnode to question concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1)  # rel from contextnode to answer concept
                        extra_j.append(_new_coord)  # contextnode coordinate
                        extra_k.append(0)  # answer concept coordinate

                half_n_rel += 2  # should be 19 now
                if len(extra_i) > 0:
                    i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                    j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                    k = torch.cat([k, torch.tensor(extra_k)], dim=0)
                ########################

                mask = (j < max_node_num) & (k < max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
                edge_type.append(i)  # each entry is [E, ]

            triple_prompt_num = tri_cls_vec.shape[0]
            triple_prompt_num = min(triple_prompt_num, max_prompt_num)
            triple_prompt_vecs[idx, :triple_prompt_num, :] = tri_cls_vec[:triple_prompt_num,:]
            triple_prompt_count[idx] = triple_prompt_num

        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type,
                         half_n_rel, triple_prompt_vecs, triple_prompt_count], f)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
          ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
          ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                      (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths, triple_prompt_vecs, triple_prompt_count = [
        x.view(-1, num_choice, *x.size()[1:])
        for x in (concept_ids, node_type_ids, node_scores, adj_lengths, triple_prompt_vecs, triple_prompt_count)
    ]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths/triple_prompt_count: (n_questions,　num_choice)
    #triple_prompt_vecs: (n_questions, num_choice, max_triple_prompt_num, triple_prompt_dim)

    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type), (triple_prompt_vecs, triple_prompt_count) #, half_n_rel * 2 + 1

def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, data, in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, :len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, :len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    (input_ids, mc_token_ids, lm_labels, mc_labels), = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels

def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)

def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length, debug_flag, dataset):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        debug_count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                if 'answerKey' in json_dic:
                    if type(json_dic['answerKey']) is bool:
                        label = int(json_dic['answerKey'])
                    elif json_dic['answerKey'].isalpha():
                        label = ord(json_dic["answerKey"]) - ord("A")
                    else:
                        print("Not implement answerKey type")
                else:
                    label = 0
                # label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
                if debug_flag:
                    debug_count += 1
                    if debug_count == 32:
                        break
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        rel_max_sl = 0
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                if dataset in ['csqav2','scitail']:
                    tokens_a = tokenizer.tokenize(context)
                    tokens_b = []
                else:
                    tokens_a = tokenizer.tokenize(context)
                    tokens_b = tokenizer.tokenize(example.question + " " + ending)

                sl = len(tokens_a) + len(tokens_b)
                rel_max_sl = max(rel_max_sl, sl)
                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                if ex_index == 0:
                    print(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            # label = label_map[example.label]
            label = example.label
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))
        print("Real Max Len: %d. Setting Max Len: %d."%(rel_max_sl, max_seq_length))
        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    try:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    except:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(model_type)

    if model_type == "roberta":
        path = "/mnt/nlp_model/huggingface/roberta-large/"
        if not os.path.exists(path):
            path = "/home/jiangjinhao/hf_models/roberta-large/"
            if not os.path.exists(path):
                path = '/mnt/jiangjinhao/hg_face/roberta-large/'
        tokenizer = tokenizer_class.from_pretrained(path)
    elif model_type == "aristo-roberta":
        tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/aristo-roberta")
    elif model_type == 'albert':
        model_path = "/mnt/nlp_model/albert-xxlarge-v2/"
        if not os.path.exists(model_path):
            model_path = "/home/jiangjinhao/hf_models/albert-xxlarge-v2/"
            if not os.path.exists(model_path):
                model_path = '/mnt/jiangjinhao/hg_face/albert-xxlarge-v2/'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert', 'aristo-roberta']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert', 'aristo-roberta'] else 1)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)

def load_bert_xlnet_roberta_input_tensors_with_triple_text(statement_jsonl_path, model_type, model_name, max_seq_length, debug_flag):
    class InputExample(object):

        def __init__(self, example_id, question, triple_evidences, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.triple_evidences = triple_evidences
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        debug_count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                triple_evidences = []
                if 'triple_evidence' in json_dic["question"]["choices"][0]:
                    for c in json_dic["question"]["choices"]:
                        triple_evi = c['triple_evidence']
                        triple_evidences.append(triple_evi)
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        triple_evidences=triple_evidences,
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
                if debug_flag:
                    debug_count += 1
                    if debug_count == 32:
                        break
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending, tri_env) in enumerate(zip(example.contexts, example.endings, example.triple_evidences)):
                tokens_a = tokenizer.tokenize(context)
                str_b = "According to: " + tri_env + " The answer is: "+ ending + "."
                # tokens_b = tokenizer.tokenize(example.question + " " + ending)
                tokens_b = tokenizer.tokenize(" "+str_b)

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_str = tokenizer.decode(input_ids, skip_special_tokens=False)
                if ex_index == 0:
                    print(input_str)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    try:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    except:
        tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(model_type)

    if model_type == "roberta":
        path = "/mnt/nlp_model/huggingface/roberta-large/"
        if not os.path.exists(path):
            path = "/home/jiangjinhao/hf_models/roberta-large/"
            if not os.path.exists(path):
                path = '/mnt/jiangjinhao/hg_face/roberta-large/'
        tokenizer = tokenizer_class.from_pretrained(path)
    elif model_type == "aristo-roberta":
        tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/aristo-roberta")
    elif model_type == 'albert':
        model_path = "/mnt/nlp_model/albert-xxlarge-v2/"
        if not os.path.exists(model_path):
            model_path = "/home/jiangjinhao/hf_models/albert-xxlarge-v2/"
            if not os.path.exists(model_path):
                model_path = '/mnt/jiangjinhao/hg_face/albert-xxlarge-v2/'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert', 'aristo-roberta']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert', 'aristo-roberta'] else 1)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)

def load_bert_xlnet_roberta_input_tensors_with_soft_prompt(statement_jsonl_path, model_type, tokenizer, max_seq_length,
                                                           prompt_data, instruct_prompt_pattern=None, debug_flag=False):
    class InputExample(object):

        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                    'prompt_mask': prompt_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask, prompt_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        debug_count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
                if debug_flag:
                    debug_count += 1
                    if debug_count == 32:
                        break
        return examples

    def convert_examples_to_features(examples, label_list, max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True,
                                     prompt_counts=None,
                                     tri_prompt_flag='[PROMPT]',
                                     ans_flag='[ANSWER]',):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        real_max_seq_len = 0
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                tri_prompt_num = prompt_counts[ex_index][ending_idx].data
                # prompt_num = instruct_prompt_num + tri_prompt_num
                tokens_a = tokenizer.tokenize(context)
                if instruct_prompt_pattern is not None:
                    # "According to:[PROMPT], the answer is[ANSWER][ENDING][ANSWER]."
                    # print(instruct_prompt_pattern)
                    # print(tokenizer.tokenize(" "+instruct_prompt_pattern))
                    ins_pro_pat_num = len(tokenizer.tokenize(" "+instruct_prompt_pattern)) - 4
                    choice = instruct_prompt_pattern.replace("[PROMPT]", "".join([tri_prompt_flag]*tri_prompt_num))
                    choice = choice.replace("[ENDING]", " " + ending)
                    # choice = choice.replace("[ANSWER]", ans_flag)
                    max_seq_length_tmp = max_seq_length + 2 # At last, this ans_flag will be removed
                else:
                    choice = ending
                    max_seq_length_tmp = max_seq_length
                # print(" " + choice)
                tokens_b = tokenizer.tokenize(" " + choice)
                # print(tokens_b)

                seq_len = len(tokens_a) + len(tokens_b)
                real_max_seq_len = max(real_max_seq_len, seq_len)

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length_tmp - special_tokens_count)

                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    B_start_idx = len(segment_ids)
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
                    B_start_idx += 1

                # print(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # print(input_ids)

                # index relevant special token
                assert len(input_ids) == len(segment_ids)
                tri_pro_tok_id = tokenizer.convert_tokens_to_ids(tri_prompt_flag)
                ans_flag_tok_id = tokenizer.convert_tokens_to_ids(ans_flag)
                tri_pro_start_idx = input_ids.index(tri_pro_tok_id, B_start_idx+1)
                tri_pro_end_idx = tri_pro_start_idx + tri_prompt_num - 1
                ans_start_idx = input_ids.index(ans_flag_tok_id)
                ans_end_idx = input_ids.index(ans_flag_tok_id, ans_start_idx+1)
                # print(tokens[B_start_idx], tokens[tri_pro_start_idx], tokens[tri_pro_end_idx], tokens[ans_start_idx], tokens[ans_end_idx])

                input_ids_new = []
                input_ids_new = input_ids_new + input_ids[0:ans_start_idx] + input_ids[ans_start_idx+1:ans_end_idx] + input_ids[ans_end_idx+1:]
                segment_ids_new = []
                segment_ids_new = segment_ids_new + segment_ids[0:ans_start_idx] + segment_ids[ans_start_idx+1:ans_end_idx] + segment_ids[ans_end_idx+1:]

                prompt_mask = []  # 0 for original token; 1 for instruct token; 2 for triple prompt token
                ori_tok_count = 0
                ins_tok_count = 0
                tri_tok_count = 0
                for i in range(len(input_ids)):
                    if i < B_start_idx:
                        prompt_mask.append(0)
                        ori_tok_count += 1
                    elif B_start_idx <= i < tri_pro_start_idx:
                        prompt_mask.append(1)
                        ins_tok_count += 1
                    elif tri_pro_start_idx <= i <= tri_pro_end_idx:
                        prompt_mask.append(2)
                        tri_tok_count += 1
                    elif tri_pro_end_idx < i < ans_start_idx or ans_end_idx < i < len(input_ids)-1:
                        prompt_mask.append(1)
                        ins_tok_count += 1
                    elif ans_start_idx < i < ans_end_idx:
                        prompt_mask.append(0)
                        ori_tok_count += 1
                    elif i == len(input_ids)-1:
                        prompt_mask.append(0)
                        ori_tok_count += 1
                assert tri_tok_count == tri_prompt_num
                # print(ins_tok_count, ins_pro_pat_num)
                assert ins_tok_count == ins_pro_pat_num
                # print(len(input_ids), len(input_ids_new), len(segment_ids_new), len(prompt_mask))
                assert len(prompt_mask) == len(input_ids_new) == len(segment_ids_new)
                # print(prompt_mask)
                input_ids = input_ids_new
                segment_ids = segment_ids_new
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # 1 for real tokens and 0 for padding tokens
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask
                    prompt_mask = ([0] * padding_length) + prompt_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    prompt_mask = prompt_mask + ([0] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(prompt_mask) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask, prompt_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        print("Real max seq len is %d, len of instruct prompt pattern %d"%(real_max_seq_len, ins_pro_pat_num))
        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_prompt_mask = torch.tensor(select_field(features, 'prompt_mask'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_prompt_mask, all_label

    # prompt_counts (num_question, num_choice)
    prompt_vecs, prompt_counts = prompt_data
    examples = read_examples(statement_jsonl_path)
    features = convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert', 'aristo-roberta']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert', 'aristo-roberta'] else 1,
                                            prompt_counts=prompt_counts)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)


def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length, debug_flag, use_triple_text, dataset):
    if model_type in ('lstm',):
        raise NotImplementedError
    elif model_type in ('gpt',):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert', 'aristo-roberta'):
        if use_triple_text:
            return load_bert_xlnet_roberta_input_tensors_with_triple_text(input_jsonl_path, model_type, model_name, max_seq_length, debug_flag)
        else:
            return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length, debug_flag, dataset)

### used for multi-process tokenize ###
def get_kl_label(label):
    label_dis = np.zeros((num_choices_global,))
    label_dis[label] = 1
    label_len = np.sum(label_dis)
    label_len = max(1,label_len)
    label_prob = np.true_divide(label_dis, label_len)
    return label_prob

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputExample(object):

    def __init__(self, example_id, question, contexts, endings, label=None):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

class InputFeatures(object):

    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'output_mask': output_mask,
            }
            for _, input_ids, input_mask, segment_ids, output_mask in choices_features
        ]
        self.label = label

def convert_examples_to_features_multiprocess(example):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    cls_token_at_end = False
    cls_token = tokenizer_global.cls_token
    sep_token = tokenizer_global.sep_token
    sep_token_extra = True
    cls_token_segment_id = 0
    pad_on_left = False  # pad on the left for xlnet
    pad_token_segment_id = 0
    sequence_a_segment_id = 0
    sequence_b_segment_id = 0
    pad_token = 0
    mask_padding_with_zero = True

    choices_features = []
    for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
        if ending_idx == 0:
            tokens_a = tokenizer_global.tokenize(context)
        tokens_b = tokenizer_global.tokenize(example.question + " " + ending)

        sl = len(tokens_a) + len(tokens_b)
        # rel_max_sl = max(rel_max_sl, sl)
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length_global - special_tokens_count)

        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer_global.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        special_token_id = tokenizer_global.convert_tokens_to_ids([cls_token, sep_token])
        output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length_global - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            output_mask = ([1] * padding_length) + output_mask

            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            output_mask = output_mask + ([1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length_global
        assert len(output_mask) == max_seq_length_global
        assert len(input_mask) == max_seq_length_global
        assert len(segment_ids) == max_seq_length_global
        choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
    label = get_kl_label(example.label)
    assert len(choices_features) == num_choices_global
    return InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label)

def load_input_tensors_for_kbqa(input_jsonl_path, model_type, model_name, max_seq_length, debug_flag, num_choices, tokenizer):
    global  tokenizer_global, max_seq_length_global, num_choices_global
    if tokenizer_global is None:
        tokenizer_global = tokenizer
    if max_seq_length_global is None:
        max_seq_length_global = max_seq_length
    if num_choices_global is None:
        num_choices_global = num_choices

    def read_examples(input_file):
        debug_count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                if 'answerKey' in json_dic:
                    label = json_dic["answerKey"]
                else:
                    label = 0
                contexts = json_dic["question"]["stem"]
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
                if debug_flag:
                    debug_count += 1
                    if debug_count == 32:
                        break
        return examples

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.float)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    examples = read_examples(input_jsonl_path)
    num_processes = 40
    with Pool(num_processes) as p:
        features = list(tqdm(p.imap(convert_examples_to_features_multiprocess, examples), total=len(examples)))
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)
    return (example_ids, all_label, *data_tensors)

def load_input_tensors_with_soft_prompt(input_jsonl_path, model_type, tokenizer, max_seq_length, prompt_data, instruct_prompt_pattern, debug_flag):
    if model_type in ('lstm',):
        raise NotImplementedError
    elif model_type in ('gpt',):
        raise NotImplementedError
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert', 'aristo-roberta'):
        return load_bert_xlnet_roberta_input_tensors_with_soft_prompt(
            input_jsonl_path, model_type, tokenizer, max_seq_length, prompt_data,
            instruct_prompt_pattern=instruct_prompt_pattern, debug_flag=debug_flag)


def load_info(statement_path: str):
    n = sum(1 for _ in open(statement_path, "r"))
    num_choice = None
    with open(statement_path, "r", encoding="utf-8") as fin:
        ids = []
        labels = []
        for line in fin:
            input_json = json.loads(line)
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            ids.append(input_json['id'])
            if num_choice is None:
                num_choice = len(input_json["question"]["choices"])
        labels = torch.tensor(labels, dtype=torch.long)

    return ids, labels, num_choice


def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict['id']
            all_dict[qid] = {
                'question': instance_dict['question']['stem'],
                'answers': [dic['text'] for dic in instance_dict['question']['choices']]
            }
    return all_dict
