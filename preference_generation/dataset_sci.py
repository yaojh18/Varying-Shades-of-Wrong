import argparse

from datasets import load_dataset
from preference_generation.utils import *


class Science(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'The final answer is ([-+]?\d+(?:,\d+)*(?:\.\d+)?(?:[eE][-+]?\d+)?)'
        self.post_process = lambda x: float(x.replace(',', ''))
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = load_dataset('xw27/scibench')
        self.dataset = []
        for data in raw_dataset['train']:
            self.dataset.append({
                'query': 'Instruction: Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal.\nQuestion: ' + data['problem_text'].strip() + f" The desired unit for your answer is: {data['unit']}.",
                'correct_answer':  float(data['answer_number'].replace(',', '').replace('−', '-'))
            })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and process answers for MMLUPro dataset')
    parser.add_argument('--dataset_name', type=str, default='Science', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='default', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='science_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=True, help='Load from existing dataset or not')

    args = parser.parse_args()

    sci_dataset = Science(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        instruction_name=args.instruction_name,
        extract_instruction_name=args.extract_instruction_name,
        response_sample_size=args.response_sample_size,
        dataset_sample_size=args.dataset_sample_size,
        load_from_exist=args.load_from_exist
    )
    sci_dataset.generate_answer()
    sci_dataset.save_dataset()
    sci_dataset.process_answer()
    sci_dataset.save_dataset()
