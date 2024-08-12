import re
import argparse
from preference_generation.utils import *


class NLGraph(RawPreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        if kwargs['dataset_name'] == 'NLGraph_shortest_path':
            self.extract_pattern = r'The total weight is (\d+)'
        elif kwargs['dataset_name'] == 'NLGraph_maximum_flow':
            self.extract_pattern = r'The maximum flow is (\d+)'
        elif kwargs['dataset_name'] == 'NLGraph_matching':
            self.extract_pattern = r'The maximum number of matches is (\d+)'
        else:
            raise NotImplementedError('NLGraph is not support. You need to extract the number manually.')
        self.map_into_index = False
        super().__init__(**kwargs)

    def load_dataset(self):
        with open(f'../dataset/{self.dataset_name}.json', 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            if self.dataset_name == 'NLGraph_shortest_path':
                pattern = r'total weight of (\d+)'
            elif self.dataset_name == 'NLGraph_maximum_flow':
                pattern = r'maximum flow.*?is (\d+)'
            elif self.dataset_name == 'NLGraph_matching':
                pattern = r'(\d+) applicants can find'
            else:
                raise NotImplementedError
            for data in dataset.values():
                correct_answer = re.search(pattern, data['answer']).group(1)
                self.dataset.append({
                    'query': 'Question: ' + data['question'][:-3],
                    'correct_answer': int(correct_answer),
                })
                if self.dataset_name == 'NLGraph_shortest_path':
                    self.dataset[-1]['query'] += ' Please also give the total weight of the shortest path.\n'
                elif self.dataset_name == 'NLGraph_matching':
                    self.dataset[-1]['query'] += ' Please also give the maximum number of matching.\n'
                else:
                    self.dataset[-1]['query'] += '\n'

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and process answers for NLGraph dataset')
    parser.add_argument('--dataset_name', type=str, default='NLGraph_shortest_path', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='llama-3', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='shortest_path_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()

    nlgraph_dataset = NLGraph(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        dataset_sample_size=args.dataset_sample_size,
        response_sample_size=args.response_sample_size,
        load_from_exist=args.load_from_exist
    )

    nlgraph_dataset.generate_answer(instruction_name=args.instruction_name)
    nlgraph_dataset.process_answer(instruction_name=args.instruction_name, extract_instruction_name=args.extract_instruction_name)
    nlgraph_dataset.save_dataset()
