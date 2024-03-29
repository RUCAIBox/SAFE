# Great Truths are Always Simple: A Rather Simple Knowledge Encoder for Enhancing the Commonsense Reasoning Capacity of Pre-Trained Models

This repo provides the source code & data of our paper: [Great Truths are Always Simple: A Rather Simple Knowledge Encoder for Enhancing the Commonsense Reasoning Capacity of Pre-Trained Models](https://arxiv.org/abs/2205.01841) (NAACL-Findings 2022).

```
@InProceedings{jiang-safe-2022,
  author =  {Jinhao Jiang, Kun Zhou, Wayne Xin Zhao and Ji-Rong Wen},
  title =   {Great Truths are Always Simple: A Rather Simple Knowledge Encoder for Enhancing the Commonsense Reasoning Capacity of Pre-Trained Models},
  year =    {2022},  
  booktitle = {North American Chapter of the Association for Computational Linguistics-Findings(NAACL-Findings)},  
}
```


<p align="center">
  <img src="./figs/overview.png" width="500" title="Overview of SAFE" alt="">
</p>
<p align="center">
  <img src="./figs/relation_feature_value_table.png" width="500" title="Values of path relation" alt="">
</p>


## Usage
### 0. Requirements

- [Python](<https://www.python.org/>) == 3.7
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.8.0
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 4.14.1
- [torch-geometric](https://pytorch-geometric.readthedocs.io/)

Run the following commands to create a conda environment (assuming CUDA-11.4):
```bash
conda create -n SAFE python=3.7
source activate SAFE
pip install numpy==1.18.3 tqdm
pip install torch==1.8.0 torchvision==0.9.0
pip install transformers==4.14.1 nltk spacy==2.3.7
python -m spacy download en

# For torch-geometric, according to the official guidlines: "You can now install PyG via Anaconda for all major OS/PyTorch/CUDA combinations 🤗 Given that you have PyTorch >= 1.8.0 installed, simply run"
conda install pyg -c pyg
```


### 1. Prepare Dataset
<!-- We strongly suggest that download the processed datasets from [Google Drive] and then diretly use them. -->
For the whole process, we mianly follow the previous work [QA-GNN](https://github.com/michiyasunaga/qagnn), and add the extra path relation extraction operation.
1. Download raw data:
```bash
./download_raw_data.sh
```
2. Preprocess the raw data by running:
```bash
python preprocess.py --run ['common', 'csqa', 'phys', 'socialiqa', 'obqa', 'obqa-fact', 'copa', 'make_word_vocab'] -p <num_processes>
python ./utils/extract_meta_path_feature.py
```
The preprocessing may take long, waiting patiently :) 

The script will:
- Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
- Convert the QA datasets into .jsonl files (e.g., stored in data/csqa/statement/)
- Identify all mentioned concepts in the questions and answers
- Extract subgraphs for each q-a pair
- Extract relation paths for each subgraph

The resulting file structure will look like:
```
.
├── README.md
├── data/
    ├── cpnet/                 (prerocessed ConceptNet)
    ├── csqa/
        ├── train_rand_split.jsonl
        ├── dev_rand_split.jsonl
        ├── test_rand_split_no_answers.jsonl
        ├── statement/             (converted statements)
        ├── grounded/              (grounded entities)
        ├── graph/                (extracted subgraphs)
            ├── train.only.qa.subgraph.adj.metapath.2.q2a.seq.pk (extracted relation path subgraphs)
        ├── ...
    ├── obqa/
    └── ...
```
### 2. Reproduce Training
For Main Results for CSQA (Table 3 in the paper), you can run:
```
./T3_csqa.sh
```
For Main Results for OBQA (Table 3 in the paper), you can run:
```
./T3_obqa.sh
```
For Supplement Results for SocialIQA, PIQA, CoPA (Table 4 in the paper), you can run:
```
./T4.sh
```
For Main Results for OBQA (Table 5 in the paper), you can run:
```
./T5.sh
```
We suggest not to change the path of dataset files. 

### Trained model checkpoint
We suggest that you can directly load the checkpoint weight to evaluate! 
For example, you can evaluate the CSQA with:
```
python3 -u qagnn.py --dataset csqa --mode 'eval_simple' --experiment_model 'lm_meta_path' --eval_detail --encoder roberta-large --inhouse True --mp_onehot_vec True --metapath_fea_hid 32 --unfreeze_epoch 4 --weight_decay 0.001 --use_score_sigmoid_mlp True --activation None --use_meta_path_feature True --mp_fc_layer 1 --dropoutmp 0.0 --dropoutf 0.0 -k 2 --fc_dim 512 --gnn_dim 32 --max_seq_len 88 --max_node_num 32 --inverse_relation False --num_relation 19 --weight_gsc 1 -elr 1e-5 -dlr 1e-2 -bs 128 --mini_batch_size 8 --eval_batch_size 8 --seed 0 --optim radam --debug False --n_epochs 30 --max_epochs_before_stop 5  --train_adj data/csqa/graph/train.onlyqa.graph.adj.metapath2.q2a.seq.pk --dev_adj  data/csqa/graph/dev.onlyqa.graph.adj.metapath2.q2a.seq.pk --test_adj data/csqa/graph/test.onlyqa.graph.adj.metapath2.q2a.seq.pk --train_statements  data/csqa/statement/train.statement.jsonl --dev_statements  data/csqa/statement/dev.statement.jsonl --test_statements  data/csqa/statement/test.statement.jsonl --load_model_path "The path of your downloaded model" 
```
<table>
  <tr>
    <th>Trained model.</th>
    <th>Dataset.</th>
    <th>Test acc.</th>
  </tr>
  <tr>
    <th>RoBERTa-large + SAFE <a href="xxx">[link]</a></th>
    <th>CSQA</th>
    <th>74.78</th>
  </tr>
  <tr>
    <th>RoBERTa-large + SAFE <a href="xxx">[link]</a></th>
    <th>OBQA</th>
    <th>70.40</th>
  </tr>
  <tr>
    <th>BERT-large + SAFE <a href="xxx">[link]</a></th>
    <th>OBQA</th>
    <th>60.20</th>
  </tr>
  <tr>
    <th>AristoRoBERTa + SAFE <a href="xxx">[link]</a></th>
    <th>OBQA</th>
    <th>87.80</th>
  </tr>
</table>

## Acknowledgment
This repo is built upon the following work:
```
QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering.
NAACL-2021
https://arxiv.org/abs/2104.06378
https://github.com/michiyasunaga/qagnn
```
Many thanks to the authors and developers!
