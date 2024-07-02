import json
from utils import *


def elicit_preference_ask_compare():
    pass


def elicit_preference_ask_score():
    pass


def elicit_preference_ppl(dataset):
    questions = []
    answers = []
    for data in dataset:
        for choice in data['choices']:
            questions.append(data['query'])
            answers.append(f'Answer: blank 1: {choice[0]}, blank 2: {choice[1]}, blank 3: {choice[2]}')

    probabilities = get_normalized_probabilities(questions, answers)
    for idx, data in enumerate(dataset):
        data['ppl'] = [probabilities[idx * 4], probabilities[idx * 4 + 1], probabilities[idx * 4 + 2],
                       probabilities[idx * 4 + 3]]

    with open('./output/MC_hard_ppl.jsonl', 'w', encoding='utf-8') as output_file:
        for data in dataset:
            output_file.write(json.dumps(data) + '\n')


def elicit_preference_consistency():
    pass
