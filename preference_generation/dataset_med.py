import argparse

from datasets import load_dataset
from preference_generation.utils import *


class MedMCQA(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = load_dataset('openlifescienceai/medmcqa')
        self.train_dataset = []
        self.test_dataset = []
        for data in raw_dataset['train']:
            self.train_dataset.append({
                'question': data['question'],
                'options': [data['opa'], data['opb'], data['opc'], data['opd']],
                'correct_answer_index':  data['cop']
            })
        for data in raw_dataset['validation']:
            self.test_dataset.append({
                'question': data['question'],
                'options': [data['opa'], data['opb'], data['opc'], data['opd']],
                'correct_answer_index': data['cop']
            })

    def precess_dataset(self, sample_size):
        random.seed(42)
        self.train_dataset = random.sample(self.train_dataset, sample_size // 5 * 4)
        self.test_dataset = random.sample(self.test_dataset, sample_size // 5)
        for data in self.train_dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            original_correctness = [0.0, 0.0, 0.0, 0.0]
            original_correctness[data['correct_answer_index']] = 1.0
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                sampled_idxs = random.sample(range(len(data['options'])), len(data['options']))
                choice = ''
                for j in range(len(data['options'])):
                    choice += f"{idx2letter[j]}. {data['options'][sampled_idxs[j]]}\n"
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query
            data['choices'] = choices
            data['correctness'] = correctness
            del data['question']
        for data in self.test_dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            original_correctness = [0.0, 0.0, 0.0, 0.0]
            original_correctness[data['correct_answer_index']] = 1.0
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                sampled_idxs = random.sample(range(len(data['options'])), len(data['options']))
                choice = ''
                for j in range(len(data['options'])):
                    choice += f"{idx2letter[j]}. {data['options'][sampled_idxs[j]]}\n"
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query
            data['choices'] = choices
            data['correctness'] = correctness
            del data['question']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and process answers for MMLUPro dataset')
    parser.add_argument('--dataset_name', type=str, default='MedMCQA', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=True, help='Load from existing dataset or not')

    args = parser.parse_args()

    med_dataset = MedMCQA(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        instruction_name=args.instruction_name,
        extract_instruction_name=args.extract_instruction_name,
        response_sample_size=args.response_sample_size,
        dataset_sample_size=args.dataset_sample_size,
        load_from_exist=args.load_from_exist
    )
    med_dataset.generate_answer()
    med_dataset.save_dataset()
    med_dataset.process_answer()
    med_dataset.save_dataset()
