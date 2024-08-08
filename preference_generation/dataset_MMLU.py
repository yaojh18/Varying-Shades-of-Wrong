import argparse
from datasets import load_dataset
from utils import *


class MMLUPro(PreferenceDataset):
    """
    Things need considering when parsing:
    1. Generated answer may not have a desired format. Need to manually add rules.
    2. Generated answer may not fall in the right index range due parser error. Need checking.
    3. Generated answer may have abstention behavior.
    """

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.filter = [
            "ori_mmlu-anatomy",
            "ori_mmlu-business_ethics",
            "ori_mmlu-college_medicine",
            "ori_mmlu-high_school_biology",
            "ori_mmlu-high_school_european_history",
            "ori_mmlu-high_school_geography",
            "ori_mmlu-high_school_government_and_politics",
            "ori_mmlu-high_school_macroeconomics",
            "ori_mmlu-high_school_microeconomics",
            "ori_mmlu-high_school_us_history",
            "ori_mmlu-high_school_world_history",
            "ori_mmlu-international_law",
            "ori_mmlu-jurisprudence",
            "ori_mmlu-logical_fallacies",
            "ori_mmlu-management",
            "ori_mmlu-moral_disputes",
            "ori_mmlu-philosophy",
            "ori_mmlu-prehistory",
            "ori_mmlu-professional_law",
            "ori_mmlu-professional_medicine",
            "ori_mmlu-security_studies",
            "ori_mmlu-sociology",
            "ori_mmlu-world_religions"
        ]
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = list(load_dataset('TIGER-Lab/MMLU-Pro')['test'])
        for data in raw_dataset:
            if data['src'] in self.filter:
                self.dataset.append({
                    'question': data['question'],
                    'options': data['options'],
                    'correct_answer_index':  data['answer_index']
                })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            original_correctness = calculate_similarity_by_ada(data['question'], data['options'], data['correct_answer_index'])
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
    parser.add_argument('--dataset_name', type=str, default='MMLUPro', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='llama-3', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='CoT', help='Name of the instruction for generating answers')
    parser.add_argument('--extract_instruction_name', type=str, default='multi_choice_extract', help='Name of the instruction for extracting answers')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()

    mmlu_dataset = MMLUPro(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        response_sample_size=args.response_sample_size,
        dataset_sample_size=args.dataset_sample_size,
        load_from_exist=args.load_from_exist
    )

    mmlu_dataset.generate_answer(instruction_name=args.instruction_name)
    mmlu_dataset.process_answer(instruction_name=args.instruction_name, extract_instruction_name=args.extract_instruction_name)
    clean_extracted_answers(mmlu_dataset, r'([A-Z])(\.|\. .+)?$')
    mmlu_dataset.save_dataset()
