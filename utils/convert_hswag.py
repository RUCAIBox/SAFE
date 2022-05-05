import json
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_hswag_statement']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_hswag_statement(qa_file: str, output_file1: str, output_file2: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file1, 'w') as output_handle1, open(output_file2, 'w') as output_handle2, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for idx, line in tqdm(enumerate(qa_handle), total=nrow):
            json_line = json.loads(line)
            label = str(json_line['label'])
            output_dict = convert_qajson_to_entailment(json_line , label)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
            output_handle2.write(json.dumps(output_dict))
            output_handle2.write("\n")
    print(f'converted statements saved to {output_file1}, {output_file2}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(line: list, label: str):
    question = line['ctx_a'].strip()
    prompt = line['ctx_b'].capitalize().strip()
    endings = line['endings']
    choice0 = prompt + " " + endings[0] if endings[0][0].isalpha() else prompt + endings[0]
    choice1 = prompt + " " + endings[1] if endings[0][0].isalpha() else prompt + endings[1]
    choice2 = prompt + " " + endings[2] if endings[0][0].isalpha() else prompt + endings[2]
    choice3 = prompt + " " + endings[3] if endings[0][0].isalpha() else prompt + endings[3]
    s0 = question + " " + choice0
    s1 = question + " " + choice1
    s2 = question + " " + choice2
    s3 = question + " " + choice3
    ans_map = {"0":"A", "1":"B", "2":"C", "3":"D"}
    dic = {'answerKey': ans_map[label],
           'id': line['ind'],
           # 'id': 0,
           'question': {'stem': question,
                        'choices': [{'label': 'A', 'text': s0}, {'label': 'B', 'text': s1},
                                    {'label': 'C', 'text': s2}, {'label': 'D', 'text': s3}]},
           'statements': [{'label': label == '0', 'statement': s0}, {'label': label == '1', 'statement': s1},
                          {'label': label == '2', 'statement': s2}, {'label': label == '3', 'statement': s3}]
           }
    return dic
