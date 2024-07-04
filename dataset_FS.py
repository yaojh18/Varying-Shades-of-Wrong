import json
import os

from utils import *


class BioGeneration(PreferenceDataset):

    def __init__(self, dataset_name, model_name, sample_size=-1, load_from_exist=False):
        super().__init__()
        random.seed(42)
        self.dataset_name = dataset_name
        self.model_name = model_name
        if load_from_exist and os.path.exists(f'./output/{self.model_name}/{self.dataset_name}.jsonl') and os.path.exists(f'./output/{self.model_name}/{self.dataset_name}_test.jsonl'):
            self.train_dataset = []
            self.test_dataset = []
            with open(f'./output/{self.model_name}/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.train_dataset.append(json.loads(line.strip()))
            with open(f'./output/{self.model_name}/{self.dataset_name}_test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.test_dataset.append(json.loads(line.strip()))
        else:
            self.load_dataset()
            self.precess_dataset(sample_size=sample_size)
            self.train_test_split()

    def load_dataset(self):
        self.dataset = []
        with open(f'./dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                self.dataset.append({
                    'query': data['input'],
                    'topic': data['topic']
                })

    def precess_dataset(self, sample_size):
        if sample_size > 0:
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)

    def save_dataset(self):
        os.makedirs(f'./output/{self.model_name}/', exist_ok=True)
        with open(f'./output/{self.model_name}/{self.dataset_name}.jsonl', 'w', encoding='utf-8') as file:
            for data in self.train_dataset:
                file.write(json.dumps(data) + '\n')
        with open(f'./output/{self.model_name}/{self.dataset_name}_test.jsonl', 'w', encoding='utf-8') as file:
            for data in self.test_dataset:
                file.write(json.dumps(data) + '\n')

    def extract_answer(self):
        """
        This function must be run separately in Python 3.7 environment.
        :return:
        """
        if torch.__version__.find('1.13.1') >= 0:
            from factscore.factscorer import FactScorer
            fs = FactScorer(
                model_name='retrieval+ChatGPT',
                openai_key=OPENAI_KEY,
                data_dir='./dataset/factscore',
                cache_dir='../.cache/factscore',
            )
            for data in tqdm(self.train_dataset, desc='Generating FactScore'):
                topics = [data['topic']] * len(data['responses'])
                data['factscore'] = fs.get_score(topics, data['responses'])['score']
        else:
            raise NotImplementedError('This function must be run separately in Pytorch==1.13.1 environment.')


if __name__ == '__main__':
    kc_dataset = BioGeneration('BioGeneration', 'gpt-4', sample_size=10, load_from_exist=True)
    # kc_dataset.generate_answer('default')
    # kc_dataset.save_dataset()
    kc_dataset.extract_answer()
    kc_dataset.save_dataset()
