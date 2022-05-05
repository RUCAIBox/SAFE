import json
import pickle
import random
import sys

import numpy as np
import os

if __name__=="__main__":
    # python ./utils/extract_few_shot.py csqa 0.2
    debug = False
    # dataset = sys.argv[1]
    # shot_ratio = float(sys.argv[2])
    # shot_ratio = 0.8 #0.2
    datasets = ['csqa','obqa']
    shot_ratios = [0.05, 0.1, 0.2, 0.5, 0.8]
    for dataset in datasets:
        for shot_ratio in shot_ratios:
            print("Start process {} {}".format(dataset,str(shot_ratio)))
            text_path = f"./data/{dataset}/statement/"
            adj_path = f"./data/{dataset}/graph/"
            in_house = True if dataset in ['csqa'] else False
            if in_house:
                inhouse_train_qids_path = f"./data/{dataset}/inhouse_split_qids.txt"

            split = "train"
            tpi = text_path + split + ".statement.jsonl"
            few_shot_idx_p = text_path + split + "." + str(shot_ratio) + ".idx"
            print("few shot idx path ",few_shot_idx_p)
            if os.path.exists(few_shot_idx_p):
                print("Load cache")
                with open(few_shot_idx_p,"rb") as fspi:
                    idx = pickle.load(fspi)
                shot = len(idx)
            else:
                print("Process few shot")
                with open(tpi, "r") as ftpi, open(few_shot_idx_p,"wb") as fspo:
                    text_line = ftpi.readlines()
                    if in_house:
                        with open(inhouse_train_qids_path, 'r') as fin:
                            inhouse_qids = set(line.strip() for line in fin)
                        shot = int(len(inhouse_qids) * shot_ratio)

                        inhouse_train_indexes = []
                        for i, line in enumerate(text_line):
                            json_dic = json.loads(line)
                            if json_dic['id'] in inhouse_qids:
                                inhouse_train_indexes.append(i)
                        idx = list(np.random.choice(inhouse_train_indexes,size=shot,replace=False))
                    else:
                        shot = int(len(text_line) * shot_ratio)
                        idx = list(np.random.choice(len(text_line), size=shot, replace=False))
                    pickle.dump(idx,fspo)

            tpo = text_path + split + ".statement.jsonl." + str(shot_ratio) + ".shot"
            api_1 = adj_path + split + ".graph.adj.pk"
            apo_1 = adj_path + split + ".graph.adj.pk." + str(shot_ratio) + ".shot"
            api_2 = adj_path + split + ".only.qa.subgraph.adj.metapath.2.q2a.seq.pk"
            apo_2 = adj_path + split + ".only.qa.subgraph.adj.metapath.2.q2a.seq.pk." + str(shot_ratio) + ".shot"
            for api, apo in zip([api_1, api_2],[apo_1, apo_2]):
                with open(tpi,"r") as ftpi, open(api,"rb") as fapi, open(tpo,"w") as ftpo, open(apo,"wb") as fapo:
                    text_line = ftpi.readlines()
                    api_line = pickle.load(fapi)
                    text_line_fs = []
                    api_line_fs = []
                    assert len(api_line) % len(text_line) == 0
                    num_c = len(api_line) / len(text_line)
                    assert len(set(idx)) == shot
                    for i in idx:
                        text_line_fs.append(text_line[i])
                        start = int(i*num_c)
                        end = int((i+1)*num_c)
                        print(i,start,end)
                        for k in range(start,end):
                            api_line_fs.append(api_line[k])
                    assert len(api_line_fs) / len(text_line_fs) == num_c
                    if debug:
                        print(text_line_fs[4])
                        print(api_line_fs[20:25])
                        break
                    ftpo.writelines(text_line_fs)
                    pickle.dump(api_line_fs,fapo)
                    print("Process %s to %s"%(api, apo))