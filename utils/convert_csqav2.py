import json
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_csqav2_statement']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_csqav2_statement(qa_file: str, output_file: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line in tqdm(qa_handle, total=nrow):
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
    print(f'converted statements saved to {output_file}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict):
    text = qa_json['question']
    id = qa_json['id']
    choice = qa_json['topic_prompt']
    statement = text

    return create_output_dict(statement, choice, qa_json.get("answer", "yes") == "yes")


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(statement: str, choice: str, label: bool) -> dict:
    dic = dict()
    dic = {'answerKey': label,
           # 'id': qa_json['id'],
           'id': 0,
           'question': {'stem': statement,
                        "choices": [{"label": "A", "text": choice}]},
           'statements': [{'label': label, 'statement': statement}]
    }
    return dic
