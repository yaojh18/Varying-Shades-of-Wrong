import re
from utils import *


class KnowledgeCrosswords(PreferenceDataset):

    def __init__(self, knowledge=True, **kwargs):
        self.knowledge = knowledge
        if self.knowledge:
            self.output_name = f"{kwargs['dataset_name']}_w_knowledge"
        else:
            self.output_name = f"{kwargs['dataset_name']}_wo_knowledge"
        super().__init__(**kwargs)

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

    def precess_dataset(self, sample_size):
        if sample_size > 0:
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            query = f'Instruction: Pick the correct answer for each blank that satisfies all the given constraints.'
            if self.knowledge:
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

    def process_answer(self):
        pattern = r'Final Answer:\s*([A-D])'
        for data in self.train_dataset:
            data['extracted answers'] = [re.search(pattern, response).group(1) if re.search(pattern, response) else None for response in data['responses']]


if __name__ == '__main__':
    kc_dataset = KnowledgeCrosswords(dataset_name='MC_hard', model_name='gpt-4', knowledge=False, sample_size=10, load_from_exist=True)
    log_probs, responses = kc_dataset.generate_answer('KC')
    kc_dataset.process_answer()
    kc_dataset.save_dataset()
