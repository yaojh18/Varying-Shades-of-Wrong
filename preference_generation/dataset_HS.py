import argparse
from datasets import load_dataset
from preference_generation.utils import *


class HellaSwag(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = list(load_dataset('Rowan/hellaswag')['train'])
        for data in raw_dataset:
            self.dataset.append({
                'context': data['ctx'],
                'options': data['endings'],
                'correct_answer_index':  int(data['label'])
            })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        instruction = 'Pick the correct completion for given context.'
        vera_model = Vera()
        for data in self.dataset:
            query = f"Instruction: {instruction}\nContext: {data['context']}\nOptions:\n"
            original_correctness = get_vera_score(vera_model, data['context'], data['options'])
            original_correctness[data['correct_answer_index']] = 1.0
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                sampled_idxs = random.sample(range(len(data['options'])), len(data['options']))
                choice = ''
                for j in range(len(data['options'])):
                    choice += f'{idx2letter[j]}. "{data["options"][sampled_idxs[j]]}"\n'
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query
            data['choices'] = choices
            data['correctness'] = correctness
            del data['context']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save answers for COM2 dataset')
    parser.add_argument('--dataset_name', type=str, default='HellaSwag', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()
    hs_dataset = HellaSwag(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        instruction_name=args.instruction_name,
        extract_instruction_name=args.extract_instruction_name,
        dataset_sample_size=args.dataset_sample_size,
        response_sample_size=args.response_sample_size,
        load_from_exist=args.load_from_exist
    )
    hs_dataset.save_dataset()
