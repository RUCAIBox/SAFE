import json
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_swag_statement']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_swag_statement(qa_file: str, output_file1: str, output_file2: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file1, 'w') as output_handle1, open(output_file2, 'w') as output_handle2, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for idx, line in tqdm(enumerate(qa_handle), total=nrow):
            line = line.strip().strip('\n')
            if idx == 0:
                continue
            else:
                content = line.split(",")
            label = content[-1]
            output_dict = convert_qajson_to_entailment(content , label)
            if idx == 0:
                print(output_dict)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
            output_handle2.write(json.dumps(output_dict))
            output_handle2.write("\n")
    print(f'converted statements saved to {output_file1}, {output_file2}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(content: list, label: str):
    question = content[4]
    prompt = content[5]
    choice0 = prompt + " " + content[7] if content[7][0].isalpha() else prompt + content[7]
    choice1 = prompt + " " + content[8] if content[7][0].isalpha() else prompt + content[7]
    choice2 = prompt + " " + content[9] if content[7][0].isalpha() else prompt + content[7]
    choice3 = prompt + " " + content[10] if content[7][0].isalpha() else prompt + content[7]
    s0 = question + " " + choice0
    s1 = question + " " + choice1
    s2 = question + " " + choice2
    s3 = question + " " + choice3
    ans_map = {"0":"A", "1":"B", "2":"C", "3":"D"}
    dic = {'answerKey': ans_map[label],
           'id': content[2],
           # 'id': 0,
           'question': {'stem': question,
                        'choices': [{'label': 'A', 'text': s0}, {'label': 'B', 'text': s1},
                                    {'label': 'C', 'text': s2}, {'label': 'D', 'text': s3}]},
           'statements': [{'label': label == '0', 'statement': s0}, {'label': label == '1', 'statement': s1},
                          {'label': label == '2', 'statement': s2}, {'label': label == '3', 'statement': s3}]
           }
    return dic
