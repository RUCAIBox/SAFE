import argparse
from utils.other_utils import bool_flag

ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
    }
}

DATASET_LIST = ['csqa', 'obqa', 'socialiqa', 'phys', 'winogrande', 'copa', 'csqav2', 'scitail','hswag','metaqa-1hop','metaqa-2hop','metaqa-3hop']

DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
    'phys': 'official',
    'winogrande': 'official',
    'copa': 'official',
    'csqav2': 'official',
    "scitail": 'official',
    "hswag": 'official',
    "metaqa-1hop": 'official',
    "metaqa-2hop": 'official',
    "metaqa-3hop": 'official',
}

DATASET_NO_TEST = ['socialiqa','phys', 'winogrande', 'csqav2', 'scitail','hswag']

EMB_PATHS = {
    'transe': 'data/transe/glove.transe.sgd.ent.npy',
    'lm': 'data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': 'data/transe/concept.nb.npy',
    'tzw': 'data/cpnet/tzw.ent.npy',
}

def add_general_arguments(parser):
    # General
    parser.add_argument('--mode', default='train', help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/{dataset}/{run_name}', help='model relevant output directory')
    parser.add_argument('--run_name', default='debug', help='the current experiment name')
    parser.add_argument('--save_model', default=True, type=bool_flag,
                        help="whether to save the latest model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--load_model_path_1', default=None)
    parser.add_argument('--load_model_path_2', default=None)
    parser.add_argument('--save_check', default=True, help='whether to save checkpoint ')
    parser.add_argument('--use_wandb', default=True, type=bool_flag, help='whether to use wandb')
    parser.add_argument("--resume_id", default=None, type=str,
                        help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")
    parser.add_argument('--continue_train_from_check_path', default=None,
                        help='path of checkpoint to continue training')


def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--ent_emb', default=['tzw'], choices=['tzw'], nargs='+', help='sources for entity embeddings')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default='csqa', choices=DATASET_LIST, help='dataset name')
    parser.add_argument('-ih', '--inhouse', required=True, type=bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--train_statements', default='data/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='data/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default=None)
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=100, type=int)
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        # inhouse=(DATASET_SETTING[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset),
                        save_dir=args.save_dir.format(dataset=args.dataset, run_name=args.run_name))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)

    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True,
                        help='use cached data to accelerate data loading')
    parser.add_argument('--log_test_pred', default=False, type=bool_flag, nargs='?', const=True,
                        help='record the prediction used to leaderboard')
    parser.add_argument('--entities_dict_path', default=None, help='The path of entities dict file')


def add_model_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    args, _ = parser.parse_known_args()
    # parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))

    # model architecture
    parser.add_argument('--experiment_model', type=str, help='experiment model, such as qagnn, gnncounter ...')
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num_in_pool', default=1, type=int, help='number of attention heads')
    parser.add_argument('--att_head_num_in_gnn', default=4, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
                        help='freeze entity embedding layer')
    parser.add_argument('--only_gnn', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether only use gnn part')
    parser.add_argument('--eval_detail', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether eval the score detail')
    parser.add_argument('--ablation', default=None, type=str, help='control the ablation study')
    parser.add_argument('--output_attentions', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether eval the attention distribution')
    parser.add_argument('--use_prompt', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether use the prompt token')
    parser.add_argument('--num_choice', default=5, type=int, help='number of choice')
    parser.add_argument('--soft_prompt_dim', default=1024, type=int, help='dim of soft prompt')
    parser.add_argument('--max_prompt_num', default=20, type=int, help='max num of soft prompt')
    parser.add_argument('--instruct_prompt_pattern', default=None, type=str,
                        help='the prompt pattern of instruction prompt')
    parser.add_argument('--drop_edge', default=None, help='the type of dropout')
    parser.add_argument('--drop_edge_ration', default=0.0, type=float, help='the ration of edge dropout')
    parser.add_argument('--use_triple_text', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether use triple text')
    parser.add_argument('--few_shot', default=False, type=bool_flag, nargs='?', const=True,
                        help='whether use few shot setting')
    parser.add_argument('--few_shot_suffix', default=None)

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')
    parser.add_argument('--do_pca', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--pca_dim', default=0, type=int, help='dimension after doing pca')
    parser.add_argument('--weight_gsc', default=1, type=int, help='weight of gsc score')
    parser.add_argument('--decouple_lm_gnn', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--scale_att_wei_to_tgt', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--self_loop', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--use_src_node_edge_count', default=True, type=bool_flag, nargs='?', const=True)
    # relation augment gnn
    parser.add_argument('--gnn_hid_dim', default=32, type=int, help='dimension of the GNN layers')
    parser.add_argument('--node_out_dim', default=32, type=int, help='dimension of the GNN layers')
    parser.add_argument('--node_in_dim', default=32, type=int, help='dimension of the GNN layers')
    parser.add_argument('--ntype_emb_dim', default=2, type=int, help='dimension of the GNN layers')
    parser.add_argument('--etype_emb_dim', default=4, type=int, help='dimension of the GNN layers')
    parser.add_argument('--nscore_emb_dim', default=2, type=int, help='dimension of the GNN layers')
    parser.add_argument('--edge_dim', default=16, type=int, help='dimension of the GNN layers')
    parser.add_argument('--inverse_relation', default=True, type=bool_flag, nargs='?', const=True)
    # meta path feature
    parser.add_argument('--use_meta_path_feature', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--metapath_fea_hid', default=32, type=int)
    parser.add_argument('--mp_fc_layer', default=0, type=int, help='number of FC layers for meta path feature')
    parser.add_argument('--softmax_mp_fea', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--mp_onehot_vec', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--use_meta_path_mlp', default=False, type=bool_flag, nargs='?', const=True,
                        help="whether directly use the mp fea count as the input to MLP")
    # meta path sa
    parser.add_argument('--metapath_fea_sa_num_head', default=3, type=int, help='number of attention heads')
    parser.add_argument('--use_score_sigmoid_mlp', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--use_score_sigmoid', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--use_score_mlp', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--sa_aggre_with_count', default=False, type=bool_flag, nargs='?', const=True)


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int, help='stop training if dev does not increase for N epochs')
    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')
    parser.add_argument('--dropoutmp', type=float, default=0.2,
                        help='dropout for fully-connected layers for meta path feature')
    parser.add_argument('--dropoutmpsa', type=float, default=0.2,
                        help='dropout for fully-connected layers for meta path feature')
    parser.add_argument('--dropoutmpscore', type=float, default=0.2,
                        help='dropout for fully-connected layers for meta path feature')
    parser.add_argument('--add_noise_flag', type=bool_flag, default=False,
                        help='flag of controlling the adding of noise')
    parser.add_argument('--plm_add_noise', type=bool_flag, default=False, help='whether use the plm and noise')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--split_optimizer', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--activation', default='gelu', type=str)
    parser.add_argument('--learn_weight_gsc', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--margin', default=0.1, type=float)


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--gpu_idx', default='0_1', type=str, help='GPU index')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')

def get_parser():
    """A helper function that handles the arguments for the whole experiment"""
    parser = argparse.ArgumentParser(add_help=False)
    add_general_arguments(parser)
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)
    if args.simple:
        parser.set_defaults(k=1)
    return parser


def get_lstm_config_from_args(args):
    lstm_config = {
        'hidden_size': args.encoder_dim,
        'output_size': args.encoder_dim,
        'num_layers': args.encoder_layer_num,
        'bidirectional': args.encoder_bidir,
        'emb_p': args.encoder_dropoute,
        'input_p': args.encoder_dropouti,
        'hidden_p': args.encoder_dropouth,
        'pretrained_emb_or_path': args.encoder_pretrained_emb,
        'freeze_emb': args.encoder_freeze_emb,
        'pool_function': args.encoder_pooler,
    }
    return lstm_config
