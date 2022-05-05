import json
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_copa_statement']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_copa_statement(qa_file: str, output_file1: str, output_file2: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file1, 'w') as output_handle1, open(output_file2, 'w') as output_handle2, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for idx, line in tqdm(enumerate(qa_handle), total=nrow):
            if idx == 0:
                continue
            else:
                content = line.split(",")
            label = content[-1].strip()
            output_dict = convert_qajson_to_entailment(content , label)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
            output_handle2.write(json.dumps(output_dict))
            output_handle2.write("\n")
    print(f'converted statements saved to {output_file1}, {output_file2}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(content: list, label: str):
    question_text = content[1]
    prompt = content[2]
    choice1 = prompt + content[3]
    choice2 = prompt + content[4]
    s1 = question_text + " " + choice1
    s2 = question_text + " " + choice2
    dic = {'answerKey': ('A' if label == '0' else 'B'),
           # 'id': qa_json['id'],
           'id': 0,
           'question': {'stem': question_text,
                        'choices': [{'label': 'A', 'text': s1}, {'label': 'B', 'text': s2}]},
           'statements': [{'label': label == '0', 'statement': s1}, {'label': label == '1', 'statement': s2}]
           }
    return dic
