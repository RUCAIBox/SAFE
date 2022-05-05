
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.convert_socialiqa import convert_to_socialiqa_statement
from utils.convert_scitail import convert_to_scitail_statement
from utils.convert_phys import convert_to_phys_statement
from utils.convert_copa import convert_to_copa_statement
from utils.conceptnet import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM, generate_adj_data_from_grounded_concepts__use_LM_mp

input_paths = {
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'obqa-fact': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'socialiqa': {
        'train': './data/socialiqa/socialiqa-train-dev/train.jsonl',
        'dev': './data/socialiqa/socialiqa-train-dev/dev.jsonl',
        'train-label': './data/socialiqa/socialiqa-train-dev/train-labels.lst',
        'dev-label': './data/socialiqa/socialiqa-train-dev/dev-labels.lst',
    },
    'phys': {
        'train': './data/phys/train.jsonl',
        'dev': './data/phys/valid.jsonl',
        'train-label': './data/phys/train-labels.lst',
        'dev-label': './data/phys/valid-labels.lst',
    },
    'copa': {
        'train': './data/copa/copa_dev_m.csv',
        'dev': './data/copa/copa_test_m.csv',
        'test': './data/copa/bcopa_ce_test_m.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'train': './data/csqa/graph/train.graph.adj.pk',
            'dev': './data/csqa/graph/dev.graph.adj.pk',
            'test': './data/csqa/graph/test.graph.adj.pk',
            'only.qa':{
                'adj-train': './data/csqa/graph/train.only.qa.subgraph.adj.pk',
                'adj-dev': './data/csqa/graph/dev.only.qa.subgraph.adj.pk',
                'adj-test': './data/csqa/graph/test.only.qa.subgraph.adj.pk',
            },
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'train': './data/obqa/graph/train.graph.adj.pk',
            'dev': './data/obqa/graph/dev.graph.adj.pk',
            'test': './data/obqa/graph/test.graph.adj.pk',
            'only.qa':{
                'adj-train': './data/obqa/graph/train.only.qa.subgraph.adj.pk',
                'adj-dev': './data/obqa/graph/dev.only.qa.subgraph.adj.pk',
                'adj-test': './data/obqa/graph/test.only.qa.subgraph.adj.pk',
            },
        },
    },
    'obqa-fact': {
        'statement': {
            'train': './data/obqa/statement/train-fact.statement.jsonl',
            'dev': './data/obqa/statement/dev-fact.statement.jsonl',
            'test': './data/obqa/statement/test-fact.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train-fact.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid-fact.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test-fact.jsonl',
        },
    },
    'socialiqa': {
        'statement': {
            'train': './data/socialiqa/statement/train.statement.jsonl',
            'dev': './data/socialiqa/statement/dev.statement.jsonl',
            'train-fairseq': './data/socialiqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/socialiqa/fairseq/official/valid.jsonl',
            'vocab': './data/socialiqa/statement/vocab.json',
        },
        'grounded': {
            'train': './data/socialiqa/grounded/train.grounded.jsonl',
            'dev': './data/socialiqa/grounded/dev.grounded.jsonl',
        },
        'graph': {
            'train': './data/socialiqa/graph/train.graph.adj.pk',
            'dev': './data/socialiqa/graph/dev.graph.adj.pk',
            'only.qa':{
                'adj-train': './data/socialiqa/graph/train.only.qa.subgraph.adj.pk',
                'adj-dev': './data/socialiqa/graph/dev.only.qa.subgraph.adj.pk',
            },
        },
    },
    'phys': {
        'statement': {
            'train': './data/phys/statement/train.statement.jsonl',
            'dev': './data/phys/statement/dev.statement.jsonl',
            'train-fairseq': './data/phys/fairseq/official/train.jsonl',
            'dev-fairseq': './data/phys/fairseq/official/valid.jsonl',
            'vocab': './data/phys/statement/vocab.json',
        },
        'grounded': {
            'train': './data/phys/grounded/train.grounded.jsonl',
            'dev': './data/phys/grounded/dev.grounded.jsonl',
        },
        'graph': {
            'train': './data/phys/graph/train.graph.adj.pk',
            'dev': './data/phys/graph/dev.graph.adj.pk',
            'only.qa':{
                'adj-train': './data/phys/graph/train.only.qa.subgraph.adj.pk',
                'adj-dev': './data/phys/graph/dev.only.qa.subgraph.adj.pk',
            },
        },
    },
    'copa': {
        'statement': {
            'train': './data/copa/statement/train.statement.jsonl',
            'dev': './data/copa/statement/dev.statement.jsonl',
            'test': './data/copa/statement/test.statement.jsonl',
            'train-fairseq': './data/copa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/copa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/copa/fairseq/official/test.jsonl',
            'vocab': './data/copa/statement/vocab.json',
        },
        'grounded': {
            'train': './data/copa/grounded/train.grounded.jsonl',
            'dev': './data/copa/grounded/dev.grounded.jsonl',
            'test': './data/copa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'train': './data/copa/graph/train.graph.adj.pk',
            'dev': './data/copa/graph/dev.graph.adj.pk',
            'test': './data/copa/graph/test.graph.adj.pk',
            'adj-train': './data/copa/graph/train.qa.subgraph.adj.pk',
            'adj-dev': './data/copa/graph/dev.qa.subgraph.adj.pk',
            'adj-test': './data/copa/graph/test.qa.subgraph.adj.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common'], choices=['common', 'csqa', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'copa', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=32, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['csqa']['graph']['train'], output_paths['cpnet']['vocab'],
                output_paths['csqa']['graph']['only.qa']['adj-train'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['csqa']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['csqa']['graph']['only.qa']['adj-dev'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                output_paths['csqa']['graph']['test'], output_paths['cpnet']['vocab'],
                output_paths['csqa']['graph']['only.qa']['adj-test'], args.nprocs, 'only.qa', args.max_node_num)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['obqa']['graph']['train'], output_paths['cpnet']['vocab'],
                output_paths['obqa']['graph']['only.qa']['adj-train'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['obqa']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['obqa']['graph']['only.qa']['adj-dev'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                output_paths['obqa']['graph']['test'], output_paths['cpnet']['vocab'],
                output_paths['obqa']['graph']['only.qa']['adj-test'], args.nprocs, 'only.qa', args.max_node_num)},
        ],

        'socialiqa':[
            {'func': convert_to_socialiqa_statement, 'args': (
                input_paths['socialiqa']['train'], input_paths['socialiqa']['train-label'],
                output_paths['socialiqa']['statement']['train'], output_paths['socialiqa']['statement']['train-fairseq'])},
            {'func': convert_to_socialiqa_statement,'args': (
                input_paths['socialiqa']['dev'], input_paths['socialiqa']['dev-label'],
                output_paths['socialiqa']['statement']['dev'], output_paths['socialiqa']['statement']['dev-fairseq'])},
            {'func': ground, 'args': (
                output_paths['socialiqa']['statement']['train'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['socialiqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (
                output_paths['socialiqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['socialiqa']['grounded']['dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['socialiqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['socialiqa']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['socialiqa']['graph']['only.qa']['adj-train'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['socialiqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['socialiqa']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['socialiqa']['graph']['only.qa']['adj-dev'], args.nprocs, 'only.qa', args.max_node_num)},

        ],

        'phys':[
            {'func': convert_to_phys_statement, 'args': (
                input_paths['phys']['train'], input_paths['phys']['train-label'],
                output_paths['phys']['statement']['train'],
                output_paths['phys']['statement']['train-fairseq'])},
            {'func': convert_to_phys_statement, 'args': (
                input_paths['phys']['dev'], input_paths['phys']['dev-label'],
                output_paths['phys']['statement']['dev'], output_paths['phys']['statement']['dev-fairseq'])},
            {'func': ground, 'args': (
                output_paths['phys']['statement']['train'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['phys']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (
                output_paths['phys']['statement']['dev'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['phys']['grounded']['dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['phys']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['phys']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['phys']['graph']['only.qa']['adj-train'], args.nprocs, 'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['phys']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['phys']['graph']['dev'], output_paths['cpnet']['vocab'],
                output_paths['phys']['graph']['only.qa']['adj-dev'], args.nprocs, 'only.qa', args.max_node_num)}
        ],

        'copa': [
            {'func': convert_to_copa_statement, 'args': (
                input_paths['copa']['train'],
                output_paths['copa']['statement']['train'], output_paths['copa']['statement']['train-fairseq'])},
            {'func': convert_to_copa_statement, 'args': (
                input_paths['copa']['dev'],
                output_paths['copa']['statement']['dev'], output_paths['copa']['statement']['dev-fairseq'])},
            {'func': convert_to_copa_statement, 'args': (
                input_paths['copa']['test'],
                output_paths['copa']['statement']['test'], output_paths['copa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (
                output_paths['copa']['statement']['train'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['copa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (
                output_paths['copa']['statement']['dev'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['copa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (
                output_paths['copa']['statement']['test'], output_paths['cpnet']['vocab'],
                output_paths['cpnet']['patterns'], output_paths['copa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['copa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['copa']['graph']['adj-train'], args.nprocs,'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['copa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['copa']['graph']['adj-dev'], args.nprocs,'only.qa', args.max_node_num)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM_mp, 'args': (
                output_paths['copa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['copa']['graph']['adj-test'], args.nprocs, 'only.qa', args.max_node_num)},
        ],

    }

    print("Begin preprocess data.")
    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
