from utils import *


class BioGeneration(PreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        super().__init__(**kwargs)

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

    def process_answer(self):
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
    kc_dataset = BioGeneration(dataset_name='BioGeneration', model_name='gpt-4', sample_size=10, load_from_exist=True)
    # kc_dataset.generate_answer('default')
    # kc_dataset.save_dataset()
    kc_dataset.process_answer()
    kc_dataset.save_dataset()
