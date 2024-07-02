import json
import re
import pandas as pd

from utils import *


class KnowledgeCrosswords(PreferenceDataset):

    def __init__(self, dataset_name, knowledge=True):
        super().__init__()
        random.seed(42)
        self.dataset_name = dataset_name
        self.knowledge = knowledge
        self.load_dataset()
        self.precess_dataset(knowledge=knowledge)

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
        # todo: debugging
        self.dataset = self.dataset[: 10]

    def precess_dataset(self, knowledge):
        idx_map = ['A', 'B', 'C', 'D']
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
                query += f'{idx_map[choice]}. '
                for blank in range(3):
                    query += f"blank {blank + 1}: {data['choice'][choice][blank]}"
                    if blank != 2:
                        query += ', '
                query += '; '
            query += '\n'
            data['query'] = query

    def save_dataset(self):
        result_df = pd.DataFrame(self.dataset)
        if self.knowledge:
            output_path = f'./output/{self.model_name}/{self.dataset_name}_w_knowledge.csv'
        else:
            output_path = f'./output/{self.model_name}/{self.dataset_name}_wo_knowledge.csv'
        result_df.to_csv(output_path, index=False, encoding='utf-8', columns=['query', 'responses', 'extracted answers', 'log_probs', 'choice', 'count'])

    @staticmethod
    def extract_answer(response_list):
        pattern = r'Final Answer:\s*([A-D])'
        answer_list = []
        for response in response_list:
            answer_list.append(re.search(pattern, response).group(1) if re.search(pattern, response) else None)
        return answer_list


if __name__ == '__main__':
    kc_dataset = KnowledgeCrosswords('MC_hard', knowledge=True)
    kc_dataset.generate_answer('gpt-4', 'KC')
    kc_dataset.save_dataset()
