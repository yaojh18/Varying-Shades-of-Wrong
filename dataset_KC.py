import json
import re
import os
import pandas as pd

from utils import *


class KnowledgeCrosswords(PreferenceDataset):

    def __init__(self, dataset_name, model_name, sample_size=-1, knowledge=True, load_from_exist=False):
        super().__init__()
        random.seed(42)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.knowledge = knowledge
        if load_from_exist and os.path.exists(f'./output/{self.model_name}/{self.dataset_name}.jsonl') and os.path.exists(f'./output/{self.model_name}/{self.dataset_name}_test.jsonl'):
            if self.knowledge:
                file_suffix = 'w_knowledge'
            else:
                file_suffix = 'wo_knowledge'
            self.train_dataset = []
            self.test_dataset = []
            with open(f'./output/{self.model_name}/{self.dataset_name}_{file_suffix}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.train_dataset.append(json.loads(line.strip()))
            with open(f'./output/{self.model_name}/{self.dataset_name}_{file_suffix}_test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.test_dataset.append(json.loads(line.strip()))
        else:
            self.load_dataset()
            self.precess_dataset(knowledge=knowledge, sample_size=sample_size)
            self.train_test_split()

    def load_dataset(self):
        self.dataset = []
        with open(f'./dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                if len(data['blanks']) == 3:
                    choices, count = generate_choices_from_candidates(data['options'], data['answer_all'])
                    self.dataset.append({
                        'source': data['source'],
                        'target': data['target'],
                        'relation': data['relation'],
                        'knowledge': data['K'],
                        'options': data['options'],
                        'choice': choices,
                        'count': count
                    })

    def precess_dataset(self, knowledge, sample_size):
        if sample_size > 0:
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            query = f'Instruction: Pick the correct answer for each blank that satisfies all the given constraints.'
            if knowledge:
                query += '\nKnowledge: '
                for k in data['knowledge'].values():
                    for term in k:
                        query += f'({term[0]}, {term[1]}, {term[2]}); '
            query += '\nConstraints: '
            for source, relation, target in zip(data['source'], data['relation'], data['target']):
                query += f'({source}, {relation}, {target}); '
            query += '\nOptions: '
            for choice in range(4):
                query += f'{idx2letter[choice]}. '
                for blank in range(3):
                    query += f"blank {blank + 1}: {data['choice'][choice][blank]}"
                    if blank != 2:
                        query += ', '
                query += '; '
            query += '\n'
            data['query'] = query

    def save_dataset(self):
        if self.knowledge:
            file_suffix = 'w_knowledge'
        else:
            file_suffix = 'wo_knowledge'
        os.makedirs(f'./output/{self.model_name}/', exist_ok=True)
        with open(f'./output/{self.model_name}/{self.dataset_name}_{file_suffix}.jsonl', 'w', encoding='utf-8') as file:
            for data in self.train_dataset:
                file.write(json.dumps(data) + '\n')
        with open(f'./output/{self.model_name}/{self.dataset_name}_{file_suffix}_test.jsonl', 'w', encoding='utf-8') as file:
            for data in self.test_dataset:
                file.write(json.dumps(data) + '\n')

    def extract_answer(self, responses_list):
        pattern = r'Final Answer:\s*([A-D])'
        for responses, data in zip(responses_list, self.train_dataset):
            data['extracted answers'] = [re.search(pattern, response).group(1) if re.search(pattern, response) else None for response in responses]


if __name__ == '__main__':
    kc_dataset = KnowledgeCrosswords('MC_hard', 'gpt-4', knowledge=True, sample_size=10)
    log_probs, responses = kc_dataset.generate_answer('KC')
    kc_dataset.extract_answer(responses)
    kc_dataset.save_dataset()
