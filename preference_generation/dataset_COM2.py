import argparse
import pandas as pd
from preference_generation.utils import *


class COM2(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = pd.read_csv(f'../dataset/{self.dataset_name}.csv', encoding='utf-8')
        for i in range(len(raw_dataset)):
            if raw_dataset.loc[i, 'label'] != 4 and (raw_dataset.loc[i, 'type'].find('2i') >= 0 or raw_dataset.loc[i, 'type'].find('3i') >= 0):
                if raw_dataset.loc[i, 'type'].find('2i') >= 0:
                    pattern = re.compile(r'What event or state is both (.*?) and also (.*?)\?')
                    match = pattern.search(raw_dataset.loc[i, 'question'])
                else:
                    pattern = re.compile(r'What event or state is both (.*?), (.*?), and also (.*?)\?')
                    match = pattern.search(raw_dataset.loc[i, 'question'])
                self.dataset.append({
                    'question': raw_dataset.loc[i, 'question'],
                    'context': raw_dataset.loc[i, 'context'],
                    'conditions': match.groups(),
                    'options': eval(raw_dataset.loc[i, 'options'])[:-1],
                    'correct_answer_index': int(raw_dataset.loc[i, 'label']),
                })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        instruction = 'As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context.'
        vera_model = Vera()
        for data in self.dataset:
            query = f"Instruction: {instruction}\nContext: {data['context']}\nQuestion: {data['question']}\nOptions:\n"
            original_correctness = get_vera_score_multihop(vera_model, data['context'], data['conditions'], data['options'])
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
            del data['context']
            del data['conditions']
        del vera_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save answers for COM2 dataset')
    parser.add_argument('--dataset_name', type=str, default='COM2', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()
    com2_dataset = COM2(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        instruction_name=args.instruction_name,
        extract_instruction_name=args.extract_instruction_name,
        dataset_sample_size=args.dataset_sample_size,
        response_sample_size=args.response_sample_size,
        load_from_exist=args.load_from_exist
    )
    com2_dataset.generate_answer()
    com2_dataset.process_answer()
    com2_dataset.save_dataset()
