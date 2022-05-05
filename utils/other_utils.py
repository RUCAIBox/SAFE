import logging
import time
import argparse
import os
import json
logger = logging.getLogger("MAIN")
import numpy as np

import torch
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling.modeling_qagnn import LM_QAGNN, LM_QAGNN_PCA_Decouple,LM_QAGNN_DataLoader,SAFE,LM_GNN_MetaPathOneHot_DataLoader, LM_QAGNN_Analysis
from utils.optimization_utils import OPTIMIZER_CLASSES

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch

def get_model(args,concept_num,concept_dim,cp_emb,mp_fea_size=None,dataset=None):
    if args.experiment_model == 'qagnn':
        logger.info("Using LM_QAGNN")
        model = LM_QAGNN(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation,
                         n_concept=concept_num,
                         concept_dim=args.gnn_dim,
                         concept_in_dim=concept_dim,
                         n_attention_head=args.att_head_num_in_pool, fc_dim=args.fc_dim,
                         n_fc_layer=args.fc_layer_num,
                         p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                         pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                         init_range=args.init_range,
                         encoder_config={'output_attentions':args.output_attentions})
    elif args.experiment_model == 'qagnn_ana':
        logger.info("Analysing LM_QAGNN")
        model = LM_QAGNN_Analysis(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation,
                         n_concept=concept_num,
                         concept_dim=args.gnn_dim,
                         concept_in_dim=concept_dim,
                         n_attention_head=args.att_head_num_in_pool, fc_dim=args.fc_dim,
                         n_fc_layer=args.fc_layer_num,
                         p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                         pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                         init_range=args.init_range,
                         encoder_config={'output_attentions':args.output_attentions})
    elif args.experiment_model == 'qagnn_decouple':
        logger.info("Using LM_QAGNN_PCA_Decouple")
        model = LM_QAGNN_PCA_Decouple(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation,
                                      n_concept=concept_num,
                                      concept_out_dim=args.gnn_dim,
                                      concept_in_dim=concept_dim,
                                      n_attention_head=args.att_head_num_in_pool, fc_dim=args.fc_dim,
                                      n_fc_layer=args.fc_layer_num,
                                      drop_emb=args.dropouti, drop_gnn=args.dropoutg, drop_fc=args.dropoutf,
                                      pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                      init_range=args.init_range,
                                      encoder_config={'output_attentions':args.output_attentions})
    elif args.experiment_model == 'lm_meta_path':
        logger.info("Using LM_GNN_Meta_Path")
        model = SAFE(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation,
                         n_concept=concept_num,
                         concept_dim=args.gnn_dim,
                         concept_in_dim=concept_dim,
                         mp_fea_size=mp_fea_size,
                         n_attention_head=args.att_head_num_in_pool, fc_dim=args.fc_dim,
                         n_fc_layer=args.fc_layer_num,
                         p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                         pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                         init_range=args.init_range,
                         encoder_config={'output_attentions':args.output_attentions})
    else:
        logger.info("Using LM_QAGNN")
        model = LM_QAGNN(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation,
                         n_concept=concept_num,
                         concept_dim=args.gnn_dim,
                         concept_in_dim=concept_dim,
                         n_attention_head=args.att_head_num_in_pool, fc_dim=args.fc_dim,
                         n_fc_layer=args.fc_layer_num,
                         p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                         pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                         init_range=args.init_range,
                         encoder_config={'output_attentions':args.output_attentions})
    return model

def get_dataloader(args, device):
    metapath_fea_size = None
    if args.use_meta_path_feature:
        dataset = LM_GNN_MetaPathOneHot_DataLoader(args, args.train_statements, args.train_adj,
                                                       args.dev_statements, args.dev_adj,
                                                       args.test_statements, args.test_adj,
                                                       batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                                       device=device,
                                                       model_name=args.encoder,
                                                       max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                                       is_inhouse=args.inhouse,
                                                       inhouse_train_qids_path=args.inhouse_train_qids,
                                                       subsample=args.subsample, use_cache=args.use_cache)
        metapath_fea_size = dataset.metapath_fea_size
    else:
        dataset = LM_QAGNN_DataLoader(args, args.train_statements, args.train_adj,
                                        args.dev_statements, args.dev_adj,
                                        args.test_statements, args.test_adj,
                                        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                        device=device,
                                        model_name=args.encoder,
                                        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                        subsample=args.subsample, use_cache=args.use_cache)
    return dataset, metapath_fea_size

def get_optimizer(model, args, dataset):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.split_optimizer:
        grouped_parameters_encoder = [
            {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.encoder_lr},
        ]
        grouped_parameters_decoder = [
            {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.decoder_lr},
        ]

        enc_optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters_encoder)
        dec_optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters_decoder)
        optimizer = (enc_optimizer, dec_optimizer)
    else:
        grouped_parameters = [
            {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.encoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
            {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.decoder_lr},
        ]
        optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)


    if args.lr_schedule == 'fixed':
        try:
            if type(optimizer) == tuple:
                enc_sche = ConstantLRSchedule(enc_optimizer)
                dec_sche = ConstantLRSchedule(dec_optimizer)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = ConstantLRSchedule(optimizer)
        except:
            if type(optimizer) == tuple:
                enc_sche = get_constant_schedule(enc_optimizer)
                dec_sche = get_constant_schedule(dec_optimizer)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            if type(optimizer) == tuple:
                enc_sche = WarmupConstantSchedule(enc_optimizer, warmup_steps=args.warmup_steps)
                dec_sche = WarmupConstantSchedule(dec_optimizer, warmup_steps=args.warmup_steps)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            if type(optimizer) == tuple:
                enc_sche = get_constant_schedule_with_warmup(enc_optimizer, num_warmup_steps=args.warmup_steps)
                dec_sche = WarmupConstantSchedule(dec_optimizer, warmup_steps=args.warmup_steps)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            if type(optimizer) == tuple:
                enc_sche = WarmupLinearSchedule(enc_optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
                dec_sche = WarmupLinearSchedule(dec_optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            if type(optimizer) == tuple:
                enc_sche = get_linear_schedule_with_warmup(enc_optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=max_steps)
                dec_sche = get_linear_schedule_with_warmup(dec_optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=max_steps)
                scheduler = (enc_sche, dec_sche)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=max_steps)

    return optimizer, scheduler

def get_dataset(args,devices):
    concept_emb = [np.load(path) for path in args.ent_emb_paths]
    concept_emb = torch.tensor(np.concatenate(concept_emb, 1), dtype=torch.float)

    concept_num, concept_dim = concept_emb.size(0), concept_emb.size(1)
    logger.info('| num_concepts: {} dim_concepts: {} |'.format(concept_num, concept_dim))

    if args.do_pca and args.pca_dim > 0:
        u, s, v = torch.pca_lowrank(A=concept_emb, q=args.pca_dim, center=True)
        concept_emb = torch.matmul(concept_emb, v)
        concept_num, concept_dim = concept_emb.size(0), concept_emb.size(1)
        logger.info('After doing PCA on concept embeddings: | num_concepts: {} dim_concepts: {} |'.format(concept_num,
                                                                                                    concept_dim))
    dataset, metapath_fea_size = get_dataloader(args, devices)

    return dataset, metapath_fea_size, concept_num, concept_dim, concept_emb

def multi_GPU_distribute(model,devices):
    # distribute the other params to device 1
    model.to(devices[1])
    # distribute the frozen concept embedding to device 0
    if hasattr(model.decoder,'concept_emb'):
        model.decoder.concept_emb.to(devices[0])

def params_statistic(model):
    pretrained_params = []
    ohter_freezed_params = []
    other_unfreezed_params = []
    safe_params = []
    for name, param in model.named_parameters():
        if 'gnn' in name:
            safe_params.append((name, param.numel(), param.device))
        if "module" not in name: # other_params
            if param.requires_grad:
                other_unfreezed_params.append((name, param.numel(), param.device))
            else:
                ohter_freezed_params.append((name, param.numel(), param.device))
        else: # pretrained_params
            pretrained_params.append((name, param.numel(), param.device))
    num_params_PLM = sum(p[1] for p in pretrained_params)
    num_params_other = sum(p[1] for p in other_unfreezed_params)
    num_params_SAFE = sum(p[1] for p in safe_params)
    logger.info('SAFE param: %f '%(num_params_SAFE))
    logger.info('Total trainable param: %d PLM param: %d  Other param: %d'%((num_params_other+num_params_PLM), num_params_PLM, num_params_other))
    for n,s,d in other_unfreezed_params:
        logger.info('  {:45}  trainable  {} device:{}'.format(n, s, d))

def print_cuda_info():
    logger.info('torch version: %s' % torch.__version__)
    logger.info('torch cuda version: %s' % torch.version.cuda)
    logger.info('cuda is available: %s'%(torch.cuda.is_available()))
    logger.info('cuda device count: %d'%(torch.cuda.device_count()))
    logger.info("cudnn version: %s"%(torch.backends.cudnn.version()))