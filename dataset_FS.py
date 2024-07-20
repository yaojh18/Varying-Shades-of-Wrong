import argparse
from utils import *


class BioGeneration(PreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        super().__init__(**kwargs)

    def load_dataset(self):
        with open(f'./dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                self.dataset.append({
                    'query': data['input'],
                    'topic': data['topic']
                })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
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
                try:
                    data['factscore'] = fs.get_score(topics, data['responses'])['score']
                except:
                    data['factscore'] = [None] * len(data['responses'])
        else:
            raise NotImplementedError('This function must be run separately in Pytorch==1.13.1 environment.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save answers for BioGeneration dataset')
    parser.add_argument('--dataset_name', type=str, default='BioGeneration', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='llama-3', help='Name of the model')
    parser.add_argument('--instruction_name', type=str, default='default', help='Name of the instruction for generating answers')
    parser.add_argument('--dataset_sample_size', type=int, default=500, help='Dataset sample size')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--load_from_exist', type=bool, default=False, help='Load from existing dataset or not')

    args = parser.parse_args()

    fs_dataset = BioGeneration(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        dataset_sample_size=args.dataset_sample_size,
        response_sample_size=args.response_sample_size,
        load_from_exist=args.load_from_exist
    )

    fs_dataset.generate_answer(args.instruction_name)
    fs_dataset.save_dataset()

    # running in separate environment
    # fs_dataset = BioGeneration(
    #     dataset_name=args.dataset_name,
    #     model_name=args.model_name,
    #     dataset_sample_size=args.dataset_sample_size,
    #     response_sample_size=args.response_sample_size,
    #     load_from_exist=args.load_from_exist
    # )
    # fs_dataset.process_answer()
    # fs_dataset.save_dataset()
