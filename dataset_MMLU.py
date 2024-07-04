import re
from datasets import load_dataset
from utils import *


class MMLUPro(PreferenceDataset):

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
        self.dataset = []
        raw_dataset = list(load_dataset('TIGER-Lab/MMLU-Pro')['test'])
        for data in raw_dataset:
            if data['src'] in self.filter:
                self.dataset.append({
                    'question': data['question'],
                    'answer': data['answer_index'],
                    'options': data['options']
                })

    def precess_dataset(self, sample_size):
        if sample_size > 0:
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            for idx, option in enumerate(data['options']):
                query += f"{idx2letter[idx]}. {option}\n"
            query += '\n'
            data['query'] = query

    def process_answer(self):
        pattern = r'Final Answer:\s*([A-Z])'
        for data in self.train_dataset:
            data['extracted answers'] = [re.search(pattern, response).group(1) if re.search(pattern, response) else None for response in data['responses']]


if __name__ == '__main__':
    mmlu_dataset = MMLUPro(dataset_name='MMLUPro', model_name='gpt-4', sample_size=10, load_from_exist=True)
    # log_probs, responses = mmlu_dataset.generate_answer('MMLU')
    mmlu_dataset.process_answer()
    mmlu_dataset.save_dataset()
