import logging
import numpy as np
import random
from tqdm import tqdm, trange
import argparse
import socket, os, subprocess, datetime, time
import json
import warnings
warnings.filterwarnings('ignore')
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import RobertaTokenizer

from utils.other_utils import freeze_net, unfreeze_net
from utils.parser_utils import get_parser
from utils.other_utils import bool_flag, export_config, check_path, get_dataloader, get_optimizer, get_model,\
    get_kbqa_dataloader
from utils import other_utils

DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
}

def get_logger(args):
    # create logger
    logger = logging.getLogger("MAIN")
    logger.setLevel(logging.DEBUG)

    # create formatter
    BASIC_FORMAT = "[%(asctime)s]-[%(levelname)s]\t[%(message)s]"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # create consle handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    # create file handler and set level to WARNING
    log_file = os.path.join(args.save_dir, "log")
    check_path(log_file)
    print("Log save to %s" % log_file)
    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger

def fix_random_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_accuracy(eval_set, model, loss_func):
    total_loss_acm_eval, n_corrects_acm_eval, n_samples_acm_eval = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            loss, n_corrects = calculate_loss(logits,labels, loss_func)
            bs = logits.shape[0]
            total_loss_acm_eval += loss.item()
            n_corrects_acm_eval += n_corrects
            n_samples_acm_eval += bs
    ave_loss_eval = total_loss_acm_eval / n_samples_acm_eval
    ave_acc_eval = n_corrects_acm_eval / n_samples_acm_eval
    return ave_loss_eval, ave_acc_eval

def calculate_loss(logits, labels, loss_func):
    bs = labels.size(0)

    if args.loss == 'margin_rank':
        n_corrects = (logits.argmax(1) == labels).sum().item()
        num_choice = logits.size(1)
        flat_logits = logits.view(-1)
        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1,num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
        wrong_logits = flat_logits[correct_mask == 0]
        y = wrong_logits.new_ones((wrong_logits.size(0),))
        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
    elif args.loss == 'cross_entropy':
        n_corrects = (logits.argmax(1) == labels).sum().item()
        loss = loss_func(logits, labels)
    elif args.loss == 'bce_cross_entropy':
        n_corrects = (logits.argmax(1) == labels).sum().item()
        logits = logits.squeeze(dim=-1)
        labels_ = labels.type_as(logits)
        loss = loss_func(logits, labels_)
    elif args.loss == 'kl_div':
        n_corrects = (logits.argmax(1) == labels).sum().item()
        answer_prob = labels.float()
        # print(answer_prob.dtype)
        logits = nn.functional.log_softmax(logits, dim=-1)
        # print(logits.dtype)
        loss = loss_func(logits, answer_prob)
    loss *= bs

    return loss, n_corrects

def main(args):
    logger.info("Fix random seed")
    fix_random_seed(args)

    config_path = os.path.join(args.save_dir, 'config.json')
    export_config(args, config_path)
    logger.info("args: {}".format(args))

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
    wandb_mode = "disabled"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.continue_train_from_check_path is not None and args.resume_checkpoint != "None"
    wandb_id = args.resume_id if resume else wandb.util.generate_id()
    args.wandb_id = wandb_id
    args.hf_version = transformers.__version__
    wandb_log = wandb.init(project="CSQA", entity="jinhao-jiang", config=args, name=args.run_name, resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode)

    logger.info(socket.gethostname())
    logger.info ("pid: %d"%(os.getpid()))
    logger.info ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
    logger.info ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
    other_utils.print_cuda_info()
    logger.info("wandb id: %s" % wandb_id)

    try: # convenient for self start of program.
        if args.mode == 'train':
            logger.info("Start training")
            train(args,wandb_log)
        elif args.mode == 'eval_simple':
            eval_simple(args)
        elif args.mode == 'eval_detail':
            eval_detail(args)
        elif args.mode == 'pred':
            pred(args)
        else:
            raise ValueError('Invalid mode')
    except RuntimeError as re:
        logger.exception(re)
        re = "{}".format(re)
        if "CUDA out of memory" in re:
            print("Out of Memory")
            exit(100)
        else:
            exit(2)
    except Exception as e:
        logger.exception(e)
        exit(1)

def train(args, wandb_log):
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt')

    ###################################################################################################
    #   Get available GPU devices                                                                     #
    ###################################################################################################
    logger.info("Available GPU devices are %s" % args.gpu_idx)
    gpu_idx_list = args.gpu_idx.split("_")
    if len(gpu_idx_list) == 2:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[1]
    else:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[0]

    device0 = torch.device(device0)
    device1 = torch.device(device1)
    devices = (device0,device1)

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    logger.info("Load dataset and dataloader")
    dataset, metapath_fea_size, concept_num, concept_dim, concept_emb = other_utils.get_dataset(args, devices)

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    logger.info("Build model")
    model = other_utils.get_model(args, concept_num, concept_dim, concept_emb, metapath_fea_size, dataset)
    other_utils.multi_GPU_distribute(model,devices)

    ###################################################################################################
    #   Build Optimizer                                                                               #
    ###################################################################################################
    logger.info("Build optimizer")
    optimizer, scheduler = other_utils.get_optimizer(model, args, dataset)

    ###################################################################################################
    #   Resume from checkpoint                                                                        #
    ###################################################################################################
    start_epoch=0
    if args.continue_train_from_check_path is not None and args.continue_train_from_check_path != 'None':
        logger.info("Resume from checkpoint %s"%args.continue_train_from_check_path)
        check = torch.load(args.continue_train_from_check_path)
        epoch = check['epoch']
        loss = check['loss']
        model_state_dict = check['model_state_dict']
        optimizer_state_dict = check['optimizer_state_dict']
        scheduler_dict = check['scheduler_dict']
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_dict)
        model.train()
        start_epoch=epoch+1

    logger.info('Parameters statistics')
    other_utils.params_statistic(model)

    logger.info('Set loss funtion of %s'%args.loss)
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=args.margin, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    elif args.loss == 'bce_cross_entropy':
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'kl_div':
        loss_func = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError("Invalid value for args.loss.")

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################
    logger.info('-' * 71)

    if args.split_optimizer:
        fre_flag = True
        unfre_flag = True
    else:
        model.train()
        freeze_net(model.encoder)
        logger.info("Freeze model.encoder")

    _, dev_acc = evaluate_accuracy(dataset.dev(), model, loss_func)
    logger.info("Before training, dev acc:{:.2f}".format(dev_acc))

    # record variables
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, best_test_acc = 0.0, 0.0, 0.0
    total_loss_acm, n_corrects_acm, n_samples_acm = 0.0, 0.0, 0.0
    best_dev_acc = dev_acc

    for epoch_id in trange(start_epoch, args.n_epochs, desc="Epoch"):
        if not args.split_optimizer:
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
                logger.info("Unfreeze model.encoder")
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
                logger.info("Freeze model.encoder")
        model.train()
        if args.plm_add_noise:
            args.add_noise_flag = True
            logger.info("add noise? %s"%args.add_noise_flag)

        start_time = time.time()
        for qids, labels, *input_data in tqdm(dataset.train(), desc="Batch"):
            if type(optimizer) is tuple:
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)

                loss, n_corrects = calculate_loss(logits,labels[a:b],loss_func)
                total_loss_acm += loss.item()

                loss = loss / bs
                loss.backward()

                n_corrects_acm += n_corrects
                n_samples_acm += (b - a)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if type(scheduler) is tuple and type(optimizer) is tuple:
                for sche in scheduler:
                    sche.step()
                enc_opt, dec_opt = optimizer
                if epoch_id < args.unfreeze_epoch:
                    if fre_flag:
                        print("Freeze model.encoder")
                        fre_flag = False
                    dec_opt.step()
                else:
                    if unfre_flag:
                        print('Unfreeze model.encoder')
                        unfre_flag = False
                    for opt in optimizer:
                        opt.step()
            else:
                optimizer.step()
                scheduler.step()

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                if type(scheduler) is tuple:
                    enc_sche, dec_sche = scheduler
                    enc_lr = enc_sche.get_last_lr()[0]
                    dec_lr = dec_sche.get_last_lr()[0]
                    logger.info('| step {:5} | enc_lr: {:9.7f} | dec_lr: {:9.7f} | train_loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, enc_lr, dec_lr, total_loss_acm / n_samples_acm, ms_per_batch))
                else:
                    logger.info('| step {:5} | lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_last_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))

                if not args.debug:
                    wandb_log.log({"lr": scheduler.get_last_lr()[0], "train_loss": total_loss_acm / n_samples_acm, "train_acc": n_corrects_acm / n_samples_acm, "ms_per_batch": ms_per_batch}, step=global_step)

                total_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                start_time = time.time()

            global_step += 1

        if args.plm_add_noise:
            args.add_noise_flag = False
            logger.info("add noise? %s"%args.add_noise_flag)

        model.eval()

        dev_acc, dev_acc = evaluate_accuracy(dataset.dev(), model, loss_func)

        test_acc = 0.0
        if args.inhouse or args.dataset in ['obqa', 'copa']:
            eval_set = dataset.test()
            total_acc = []
            count = 0
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(eval_set):
                        count += 1
                        logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                        predictions = logits.argmax(1) #[bsize, ]
                        preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                        for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                            acc = int(pred.item()==label.item())
                            print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            f_preds.flush()
                            total_acc.append(acc)
            test_acc = float(sum(total_acc))/len(total_acc)
        elif not args.inhouse and epoch_id > 10 and args.dataset in ['csqa']: # leaderboard format
            test_set = dataset.test()
            suffix = 'leaderboard_preds_' + str(epoch_id) + '.csv'
            preds_path = os.path.join(args.save_dir, suffix)
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(test_set):
                        logits, _ = model(*input_data, detail=False)
                        predictions = logits.argmax(1)  # [bsize, ]
                        for qid, pred in zip(qids, predictions):
                            print('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            f_preds.flush()
            logger.info("Write the leaderboard prediction of %s in to %s" % (model_path, preds_path))

        best_test_acc = max(test_acc, best_test_acc)
        if epoch_id > args.unfreeze_epoch:
            # update record variables
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    model_path = os.path.join(args.save_dir, 'model.pt')
                    torch.save([model.state_dict(), args], model_path)
                    logger.info("model saved to %s"%model_path)
        else:
            best_dev_epoch = epoch_id

        logger.info('-' * 71)
        logger.info(
            '| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc,
                                                                                     test_acc))
        logger.info('| best_dev_epoch {:3} | best_dev_acc {:7.4f} | final_test_acc {:7.4f} |'.format(best_dev_epoch,
                                                                                                     best_dev_acc,
                                                                                                     final_test_acc))
        logger.info('-' * 71)

        if not args.debug:
            wandb_log.log({"dev_acc": dev_acc, "dev_loss": dev_acc, "best_dev_acc": best_dev_acc,
                           "best_dev_epoch": best_dev_epoch}, step=global_step)
            if test_acc > 0:
                wandb_log.log({"test_acc": test_acc, "test_loss": 0.0, "final_test_acc": final_test_acc},
                          step=global_step)

        if args.save_check:
            training_dict = {'epoch':epoch_id, 'loss':loss,
                             'model_state_dict':model.state_dict(),
                             'optimizer_state_dict':optimizer.state_dict(),
                             'scheduler_dict':scheduler.state_dict()}
            torch.save(training_dict, checkpoint_path)


        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            logger.info("After %d epoch no improving. Stop!"%(epoch_id-best_dev_epoch))
            logger.info("Best test accuracy: %s"%str(best_test_acc))
            logger.info("Final best test accuracy according to dev: %s"%str(final_test_acc))
            if args.eval_detail:
                logger.info("Output evaluate details")
                args.load_model_path = model_path
                eval_detail(args)
            break

        model.train()

def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    print('inhouse?', args.inhouse)
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} dim_concepts: {} |'.format(concept_num, concept_dim))

    if args.do_pca and args.pca_dim > 0:
        u, s, v = torch.pca_lowrank(A=cp_emb, q=args.pca_dim, center=True)
        cp_emb = torch.matmul(cp_emb, v)
        concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
        print('After doing PCA on concept embeddings: | num_concepts: {} dim_concepts: {} |'.format(concept_num,
                                                                                                    concept_dim))

    gpu_idx_list = args.gpu_idx.split("_")
    if len(gpu_idx_list) == 2:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[1]
    else:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[0]

    device0 = torch.device(device0)
    device1 = torch.device(device1)
    device = (device0, device1)

    dataset, metapath_fea_size = get_dataloader(args, device)

    model_state_dict, old_args = torch.load(model_path)
    model = get_model(args, concept_num, concept_dim, cp_emb, metapath_fea_size, dataset)
    model.load_state_dict(model_state_dict)
    # model.decoder.weight_gsc = 2

    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    with open(args.dev_statements, "r", encoding="utf-8") as f:
        examples = {}
        for line in f.readlines():
            json_dic = json.loads(line)
            qid = json_dic['id']
            choices = json_dic["question"]["choices"]
            choice_str = "\t".join([ending['label']+':'+ending['text'] for ending in choices])
            choice_str = choice_str + "\n" + "Answer: " + json_dic["answerKey"]
            question = json_dic["question"]["stem"]
            question = question+'\n'+choice_str
            examples[qid] = question

    eval_set = dataset.dev()
    total_acc = []
    cs_acc = []
    gs_acc = []
    help_by_gs_count = 0
    mislead_by_gs_count = 0
    count = 0
    preds_path = os.path.join(args.save_dir, 'dev_pred_details.txt')
    i2c = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E'}
    with open(preds_path, 'w') as f_preds:
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(eval_set):
                logits, scores = model(*input_data, detail=False)
                context_scores, graph_scores = scores
                predictions = logits.argmax(1) #[bsize, ]
                preds_ranked = (-logits).argsort(1) #[bsize, n_choices] descending rank
                for i, (qid, label, pred, _preds_ranked, qa_score, graph_score) in \
                    enumerate(zip(qids, labels, predictions, preds_ranked, context_scores, graph_scores)):
                    count += 1
                    acc = int(pred.item() == label.item())
                    _cs_acc = int(qa_score.argmax(-1).item() == label.item())
                    _gs_acc = int(graph_score.argmax(-1).item() == label.item())
                    flag = ''
                    if _cs_acc==1 and _gs_acc==1:
                        flag = 'both judge right\n'
                    elif _cs_acc==0 and _gs_acc==0 and acc==0:
                        flag = 'both judge false\n'
                    elif _cs_acc == 0 and _gs_acc == 0 and acc == 1:
                        flag = 'both judge false, but merge is right\n'
                    elif _cs_acc==1 and _gs_acc==0 and acc==1:
                        flag = 'extra feature judge false, but not influence PLM\n'
                    elif _cs_acc==1 and _gs_acc==0 and acc==0:
                        mislead_by_gs_count += 1
                        flag = "mislead by extra feature\n"
                    elif _cs_acc==0 and _gs_acc==1 and acc==1:
                        help_by_gs_count += 1
                        flag = 'help by extra feature\n'
                    elif _cs_acc==0 and _gs_acc==1 and acc==0:
                        flag = 'PLM judge false, but not be recorrected by extra feature\n'


                    question = examples[qid] + '\n'
                    _preds_ranked = "\t".join([i2c[str(i)] for i in _preds_ranked.cpu().numpy().tolist()])
                    _preds_ranked = "Prediction rank:\t" + _preds_ranked + '\n'
                    qa_score = "QA context score:\t" + '\t'.join([str(round(i,2)) for i in qa_score.cpu().numpy().tolist()]) + '\n'
                    graph_score = "Graph score:\t" + '\t'.join([str(round(i,2)) for i in graph_score.cpu().numpy().tolist()]) + '\n'
                    case = question + flag + _preds_ranked + qa_score + graph_score + '\n---------------------------------\n'
                    f_preds.write(case)
                    total_acc.append(acc)
                    cs_acc.append(_cs_acc)
                    gs_acc.append(_gs_acc)

    assert len(total_acc) == len(cs_acc) == len(gs_acc)
    dev_acc_with_both = float(sum(total_acc))/len(total_acc)
    dev_acc_with_cs = float(sum(cs_acc))/len(cs_acc)
    dev_acc_with_gs = float(sum(gs_acc))/len(gs_acc)
    print('-' * 71)
    print('dev_acc_with_both {:7.4f}'.format(dev_acc_with_both))
    print('dev_acc_with_only_cs {:7.4f}'.format(dev_acc_with_cs))
    print('dev_acc_with_only_gs {:7.4f}'.format(dev_acc_with_gs))
    print('help_by_extra_feature: {:d}/{:d} {:7.4f}'.format(help_by_gs_count, count, float(help_by_gs_count)/count))
    print('mislead_by_extra_feature: {:d}/{:d} {:7.4f}'.format(mislead_by_gs_count, count, float(mislead_by_gs_count)/count))
    print('-' * 71)

def pred(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    print('inhouse?', args.inhouse)
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    gpu_idx_list = args.gpu_idx.split("_")
    if len(gpu_idx_list) == 2:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[1]
    else:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[0]

    device0 = torch.device(device0)
    device1 = torch.device(device1)
    device = (device0, device1)

    dataset, metapath_fea_size = get_dataloader(args, device)

    model_state_dict, old_args = torch.load(model_path)
    model = get_model(args, concept_num, concept_dim, cp_emb, metapath_fea_size, dataset)
    model.load_state_dict(model_state_dict)

    model.encoder.to(device0)
    model.decoder.to(device1)

    test_set = dataset.test()
    preds_path = os.path.join(args.save_dir, 'preds.csv')
    with open(preds_path, 'w') as f_preds:
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(test_set):
                logits, _ = model(*input_data)
                predictions = logits.argmax(1)  # [bsize, ]
                for qid, pred in zip(qids, predictions):
                    print('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                    f_preds.flush()
    print("Write the prediction of %s in to %s" % (model_path, preds_path))

def eval_detail_with_two_model(args):
    assert args.load_model_path_1 is not None and args.load_model_path_2 is not None
    model1_path = args.load_model_path_1
    model2_path = args.load_model_path_2
    print('inhouse?', args.inhouse)
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    gpu_idx_list = args.gpu_idx.split("_")
    if len(gpu_idx_list) == 2:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[1]
    else:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[0]

    device0 = torch.device(device0)
    device1 = torch.device(device1)
    device = (device0, device1)

    dataset, metapath_fea_size = get_dataloader(args, device)

    model1_state_dict, old1_args = torch.load(model1_path)
    model1 = get_model(old1_args, concept_num, concept_dim, cp_emb, metapath_fea_size)
    model1.load_state_dict(model1_state_dict)
    model1.encoder.to(device0)
    model1.decoder.to(device1)
    model1.eval()

    model2_state_dict, old2_args = torch.load(model2_path)
    model2 = get_model(old2_args, concept_num, concept_dim, cp_emb, metapath_fea_size)
    model2.load_state_dict(model2_state_dict)
    model2.encoder.to(device0)
    model2.decoder.to(device1)
    model2.eval()


    with open(args.dev_statements, "r", encoding="utf-8") as f:
        examples = {}
        for line in f.readlines():
            json_dic = json.loads(line)
            qid = json_dic['id']
            choices = json_dic["question"]["choices"]
            choice_str = "\t".join([ending['label']+':'+ending['text'] for ending in choices])
            choice_str = choice_str + "\n" + "Answer: " + json_dic["answerKey"]
            question = json_dic["question"]["stem"]
            question = question+'\n'+choice_str
            examples[qid] = question

    eval_set = dataset.dev()
    total_acc1 = []
    total_acc2 = []
    cs_acc1 = []
    cs_acc2 = []
    gs_acc2 = []
    help_by_GNN_count = 0
    mislead_by_GNN_count = 0
    count = 0
    preds_path = os.path.join(args.save_dir, 'dev_pred_details.txt')
    i2c = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E'}
    with open(preds_path, 'w') as f_preds:
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(eval_set):
                logits_1, scores_1 = model1(*input_data, detail=False)
                context_scores_1, graph_scores_1 = scores_1
                predictions_1 = logits_1.argmax(1) #[bsize, ]
                preds_ranked_1 = (-logits_1).argsort(1) #[bsize, n_choices] descending rank
                logits_2, scores_2 = model2(*input_data, detail=False)
                context_scores_2, graph_scores_2 = scores_2
                predictions_2 = logits_2.argmax(1)  # [bsize, ]
                preds_ranked_2 = (-logits_2).argsort(1)  # [bsize, n_choices] descending rank
                for i, (qid, label, pred1, _preds_ranked1, qa_score1, graph_score1,
                                    pred2, _preds_ranked2, qa_score2, graph_score2 ) in \
                    enumerate(zip(qids, labels, predictions_1, preds_ranked_1, context_scores_1, graph_scores_1,
                                                predictions_2, preds_ranked_2, context_scores_2, graph_scores_2)):
                    count += 1
                    acc1 = int(pred1.item() == label.item())
                    acc2 = int(pred2.item() == label.item())
                    _cs_acc1 = int(qa_score1.argmax(-1).item() == label.item())
                    _cs_acc2 = int(qa_score2.argmax(-1).item() == label.item())
                    _gs_acc1 = int(graph_score1.argmax(-1).item() == label.item())
                    _gs_acc2 = int(graph_score2.argmax(-1).item() == label.item())
                    flag = ''
                    if acc1 == 1 and acc2 == 1:
                        if _cs_acc2==1 and _gs_acc2==1:
                            flag = 'Only PLM or PLM + GNN all right / Both judge right\n'
                        elif _cs_acc2==1 and _gs_acc2==0:
                            flag = 'Only PLM or PLM + GNN all right / PLM not be mislead by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==1:
                            flag = 'Only PLM or PLM + GNN all right / PLM helped by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==0:
                            flag = 'Only PLM or PLM + GNN all right / PLM and GNN both false, but make a right prediction\n'
                    elif acc1 == 0 and acc2 == 1:
                        help_by_GNN_count += 1
                        if _cs_acc2==1 and _gs_acc2==1:
                            flag = 'PLM helped by GNN / Both judge right\n'
                        elif _cs_acc2==1 and _gs_acc2==0:
                            flag = 'PLM helped by GNN / PLM not be mislead by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==1:
                            flag = 'PLM helped by GNN / PLM helped by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==0:
                            flag = 'PLM helped by GNN / PLM and GNN both false, but make a right prediction\n'
                    elif acc1 == 1 and acc2 == 0:
                        mislead_by_GNN_count += 1
                        if _cs_acc2==1 and _gs_acc2==0:
                            flag = 'PLM mislead by GNN / PLM be mislead by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==1:
                            flag = 'PLM mislead by GNN / PLM not be helped by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==0:
                            flag = 'PLM mislead by GNN / PLM and GNN both false, so make a false prediction\n'
                    elif acc1 == 0 and acc2 == 0:
                        if _cs_acc2==1 and _gs_acc2==0:
                            flag = 'Only PLM or PLM + GNN all false / PLM be mislead by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==1:
                            flag = 'Only PLM or PLM + GNN all false / PLM not be helped by GNN\n'
                        elif _cs_acc2==0 and _gs_acc2==0:
                            flag = 'Only PLM or PLM + GNN all false / PLM and GNN both false, so make a false prediction\n'

                    question = examples[qid] + '\n'
                    # _preds_ranked = "\t".join([i2c[str(i)] for i in _preds_ranked.cpu().numpy().tolist()])
                    # _preds_ranked = "Prediction rank:\t" + _preds_ranked + '\n'
                    plm_score = "Only PLM score:\t" + '\t'.join([str(round(i,2)) for i in qa_score1.cpu().numpy().tolist()]) + '\n'
                    qa_score = "PLM+GNN score:\n" + "\tQA context score:\t" + '\t'.join([str(round(i,2)) for i in qa_score2.cpu().numpy().tolist()]) + '\n'
                    graph_score = "\tGraph score:\t" + '\t'.join([str(round(i,2)) for i in graph_score2.cpu().numpy().tolist()]) + '\n'
                    # case = question + flag + _preds_ranked + qa_score + graph_score + '\n---------------------------------\n'
                    case = question + flag + plm_score + qa_score + graph_score + '\n---------------------------------\n'
                    f_preds.write(case)
                    total_acc1.append(acc1)
                    cs_acc1.append(_cs_acc1)
                    total_acc2.append(acc2)
                    cs_acc2.append(_cs_acc2)
                    gs_acc2.append(_gs_acc2)

    assert len(total_acc1) == len(cs_acc1) == len(gs_acc2) == len(total_acc2) == len(cs_acc2)
    dev_acc_with_both_1 = float(sum(total_acc1))/len(total_acc1)
    dev_acc_with_cs_1 = float(sum(cs_acc1))/len(cs_acc1)
    dev_acc_with_both_2 = float(sum(total_acc2))/len(total_acc2)
    dev_acc_with_cs_2 = float(sum(cs_acc2))/len(cs_acc2)
    dev_acc_with_gs_2 = float(sum(gs_acc2))/len(gs_acc2)
    print('-' * 71)
    print("Only PLM:")
    print('dev_acc {:7.4f}'.format(dev_acc_with_both_1))
    print("PLM+GNN:")
    print('dev_acc_with_both {:7.4f}'.format(dev_acc_with_both_2))
    print('dev_acc_with_only_cs {:7.4f}'.format(dev_acc_with_cs_2))
    print('dev_acc_with_only_gs {:7.4f}'.format(dev_acc_with_gs_2))
    print("Compare two model:")
    print('help_by_gnn: {:d}/{:d} {:7.4f}'.format(help_by_GNN_count, count, float(help_by_GNN_count)/count))
    print('mislead_by_gnn: {:d}/{:d} {:7.4f}'.format(mislead_by_GNN_count, count, float(mislead_by_GNN_count)/count))
    print('-' * 71)

def eval_simple(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    print('inhouse?', args.inhouse)
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    gpu_idx_list = args.gpu_idx.split("_")
    if len(gpu_idx_list) == 2:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[1]
    else:
        device0 = 'cuda:' + gpu_idx_list[0]
        device1 = 'cuda:' + gpu_idx_list[0]

    device0 = torch.device(device0)
    device1 = torch.device(device1)
    device = (device0, device1)

    dataset, metapath_fea_size = get_dataloader(args, device)

    model_state_dict, old_args = torch.load(model_path)
    model = get_model(args, concept_num, concept_dim, cp_emb, metapath_fea_size, dataset)
    model.load_state_dict(model_state_dict)

    model.encoder.to(device0)
    model.decoder.to(device1)

    eval_set = dataset.dev()
    test_set = dataset.test()

    dev_acc = evaluate_accuracy(eval_set, model)
    test_acc = evaluate_accuracy(test_set, model)
    print("dev_acc: ", dev_acc)
    print("test_acc: ", test_acc)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger(args)
    main(args)
