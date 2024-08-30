import argparse
from preference_generation.utils import *


class KnowledgeCrosswords(RawPreferenceDataset):
    """
    Fix correct answer number to 3 only. Easier for future comparison
    """

    def __init__(self, knowledge=True, **kwargs):
        self.knowledge = knowledge
        if self.knowledge:
            self.output_name = f"{kwargs['dataset_name']}_w_knowledge"
        else:
            self.output_name = f"{kwargs['dataset_name']}_wo_knowledge"
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.map_into_index = True
        super().__init__(**kwargs)

    def load_dataset(self):
        with open(f'../dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                if len(data['blanks']) == 3:
                    self.dataset.append({
                        'source': data['source'],
                        'target': data['target'],
                        'relation': data['relation'],
                        'knowledge': data['K'],
                        'options': data['options'],
                        'correct_answer_index': data['answer_all']
                    })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            original_choices, original_correctness = generate_choices_from_candidates(data['options'], data['correct_answer_index'])
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
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                choice = ''
                sampled_idxs = random.sample(range(4), 4)
                for j in range(4):
                    choice += f'{idx2letter[j]}. '
                    for k in range(3):
                        choice += f"blank {k + 1}: {original_choices[sampled_idxs[j]][k]}"
                        if k != 2:
                            choice += ', '
                        else:
                            choice += '; '
                choice += '\n\n'
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query
            data['correctness'] = correctness
            data['choices'] = choices
            del data['source']
            del data['relation']
            del data['target']
            del data['knowledge']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and process answers for KnowledgeCrosswords dataset')
    parser.add_argument('--dataset_name', type=str, default='KC', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--knowledge', type=bool, default=False, help='Include knowledge or not')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()

    kc_dataset = KnowledgeCrosswords(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        knowledge=args.knowledge,
        instruction_name=args.instruction_name,
        extract_instruction_name=args.extract_instruction_name,
        response_sample_size=args.response_sample_size,
        dataset_sample_size=args.dataset_sample_size,
        load_from_exist=args.load_from_exist
    )

    kc_dataset.generate_answer()
    kc_dataset.process_answer()
    kc_dataset.save_dataset()
