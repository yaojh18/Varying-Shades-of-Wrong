import argparse
from preference_generation.utils import *


class BioGeneration(RawPreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        super().__init__(**kwargs)

    def load_dataset(self):
        with open(f'../dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
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

    def process_answer(self, split='train', key=None):
        """
        This function must be run separately in Python 3.7 environment.
        :return:
        """
        if torch.__version__.find('1.13.1') >= 0:
            from factscore.factscorer import FactScorer
            fs = FactScorer(
                model_name='retrieval+ChatGPT',
                openai_key=OPENAI_KEY,
                data_dir='../dataset/factscore',
                cache_dir='../.cache/factscore',
            )
            responses_name = 'responses' if key is None else key + '_responses'
            factscore_name = 'factscore' if key is None else key + '_factscore'

            for data in tqdm(self.train_dataset if split == 'train' else self.test_dataset, desc='Generating FactScore'):
                if 'factscore' not in data:
                    with ProcessPoolExecutor(max_workers=2) as executor:
                        futures = [executor.submit(self.get_factscore_for_single_response, index, fs, data['topic'], response)
                                   for index, response in enumerate(data[responses_name])]
                        factscore_dict = collections.defaultdict(str)
                        for job in as_completed(futures):
                            index, factscore = job.result(timeout=None)
                            factscore_dict[index] = factscore
                        data[factscore_name] = [factscore_dict[i] for i in range(len(data[responses_name]))]
                        self.save_dataset()
        else:
            raise NotImplementedError('This function must be run separately in Pytorch==1.13.1 environment.')

    @staticmethod
    def get_factscore_for_single_response(idx, factscorer, topic, response):
        try:
            return idx, factscorer.get_score([topic], [response])['score']
        except Exception as e:
            print(e)
            return idx, None


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Generate and save answers for BioGeneration dataset')
    # parser.add_argument('--dataset_name', type=str, default='BioGeneration_test', help='Name of the dataset')
    # parser.add_argument('--model_name', type=str, default='gpt-3.5', help='Name of the model')
    # parser.add_argument('--instruction_name', type=str, default='default', help='Name of the instruction for generating answers')
    # parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    # parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    # parser.add_argument('--load_from_exist', type=bool, default=True, help='Load from existing dataset or not')
    #
    # args = parser.parse_args()
    #
    # fs_dataset = BioGeneration(
    #     dataset_name=args.dataset_name,
    #     model_name=args.model_name,
    #     dataset_sample_size=args.dataset_sample_size,
    #     response_sample_size=args.response_sample_size,
    #     load_from_exist=args.load_from_exist
    # )
    #
    # fs_dataset.generate_answer(args.instruction_name)
    # fs_dataset.save_dataset()
    torch.multiprocessing.set_start_method('spawn')

    fs_dataset = BioGeneration(
        dataset_name='BioGeneration',
        model_name='llama-3',
        load_from_exist=True
    )
    fs_dataset.process_answer()

    fs_dataset = BioGeneration(
        dataset_name='BioGeneration',
        model_name='gpt-4',
        load_from_exist=True
    )
    fs_dataset.process_answer()
