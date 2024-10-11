import argparse
import json
import os
import chess
import datasets
import pandas as pd
from chess import Board, Move
from chess.engine import SimpleEngine
from typing import Callable
from torch.utils.data import random_split

from utils import *


class RawPreferenceDataset:

    dataset_name: str
    model_name: str
    extract_pattern: str
    post_process: Callable
    output_name: str = ''
    split_ratio: float = 0.8

    def __init__(self,
                 dataset_name,
                 model_name,
                 instruction_name,
                 extract_instruction_name,
                 dataset_sample_size=-1,
                 response_sample_size=10,
                 load_from_exist=False,
                 load_test_path=None
                 ):
        random.seed(42)
        self.dataset = []
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.instruction_name = instruction_name
        self.extract_instruction_name = extract_instruction_name
        self.dataset_sample_size = dataset_sample_size
        self.response_sample_size = response_sample_size
        self.load_test_path = load_test_path

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

        if self.output_name == '':
            self.output_name = self.dataset_name
        if load_from_exist and load_test_path is not None and os.path.exists(load_test_path):
            self.train_dataset = []
            self.test_dataset = []
            with open(load_test_path, 'r', encoding='utf-8') as file:
                for line in file:
                    self.test_dataset.append(json.loads(line.strip()))
        elif load_from_exist and os.path.exists(f'../output/{self.model_name}/{self.output_name}.jsonl'):
            self.train_dataset = []
            self.test_dataset = []
            with open(f'../output/{self.model_name}/{self.output_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.train_dataset.append(json.loads(line.strip()))
        else:
            self.load_dataset()
            self.precess_dataset(sample_size=self.dataset_sample_size)
            self.train_test_split()

    def load_dataset(self):
        pass

    def precess_dataset(self, sample_size):
        pass

    def train_test_split(self):
        if self.train_dataset is None and self.test_dataset is None:
            train_dataset_size = round(len(self.dataset) * self.split_ratio)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            self.train_dataset, self.test_dataset = random_split(self.dataset, [train_dataset_size, len(self.dataset) - train_dataset_size])
            self.train_dataset = list(self.train_dataset)
            self.test_dataset = list(self.test_dataset)

    def generate_answer(self, split='train', key=None, peft_dir=None):
        with open(f'../instruction/{self.instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        queries = []
        for data in self.train_dataset if split == 'train' else self.test_dataset:
            if 'choices' in data:
                for choice in data['choices']:
                    queries.append([{'role': 'user', 'content': instruction + data['query'] + choice}])
            else:
                queries += [[{'role': 'user', 'content': instruction + data['query']}] for _ in range(self.response_sample_size)]
        if self.model_name == 'gpt-4':
            log_probs, responses = batch_query_openai(queries, model_name='gpt-4o')
        elif self.model_name == 'gpt-3.5':
            log_probs, responses = batch_query_openai(queries, model_name='gpt-3.5-turbo')
        elif self.model_name == 'llama-3':
            log_probs, responses = batch_query_open_sourced_llm(queries, model_name='meta-llama/Meta-Llama-3-8B-Instruct')
        elif peft_dir is not None:
            log_probs, responses = batch_query_open_sourced_llm(
                queries,
                model_name='meta-llama/Meta-Llama-3-8B-Instruct',
                peft_dir=peft_dir
            )
        else:
            raise NotImplementedError
        responses_name = 'responses' if key is None else key + '_responses'
        log_probs_name = 'log_probs' if key is None else key + '_log_probs'
        for i, data in enumerate(self.train_dataset if split == 'train' else self.test_dataset):
            data[log_probs_name] = log_probs[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]
            data[responses_name] = responses[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]

    def process_answer(self, split='train', key=None, peft_dir=None):
        with open(f'../instruction/{self.instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        with open(f'../instruction/{self.extract_instruction_name}.txt', encoding='utf-8') as f:
            extract_instruction = ''.join(f.readlines())
        queries = []
        responses_name = 'responses' if key is None else key + '_responses'
        for data in self.train_dataset if split == 'train' else self.test_dataset:
            if 'choices' in data:
                for choice, response in zip(data['choices'], data[responses_name]):
                    queries.append([
                        {'role': 'user', 'content': instruction + data['query'] + choice},
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': extract_instruction}
                    ])
            else:
                for response in data[responses_name]:
                    queries.append([
                        {'role': 'user', 'content': instruction + data['query']},
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': extract_instruction}
                    ])
        if self.model_name == 'gpt-4':
            _, responses = batch_query_openai(queries, model_name='gpt-4o', mode='extract')
        elif self.model_name == 'gpt-3.5':
            _, responses = batch_query_openai(queries, model_name='gpt-3.5-turbo', mode='extract')
        elif self.model_name == 'llama-3':
            _, responses = batch_query_open_sourced_llm(queries, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='extract')
        elif peft_dir is not None:
            log_probs, responses = batch_query_open_sourced_llm(
                queries,
                model_name='meta-llama/Meta-Llama-3-8B-Instruct',
                peft_dir=peft_dir,
                mode='extract'
            )
        else:
            raise NotImplementedError
        extracted_answers_name = 'extracted_answers' if key is None else key + '_extracted_answers'
        for i, data in enumerate(self.train_dataset if split == 'train' else self.test_dataset):
            data[extracted_answers_name] = responses[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]
        clean_extracted_answers(
            dataset=self.train_dataset if split == 'train' else self.test_dataset,
            key=extracted_answers_name,
            pattern=self.extract_pattern,
            post_process=self.post_process
        )

    def save_dataset(self):
        if len(self.train_dataset) > 0:
            os.makedirs(f'../output/{self.model_name}/', exist_ok=True)
            with open(f'../output/{self.model_name}/{self.output_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in self.train_dataset:
                    file.write(json.dumps(data) + '\n')
        if len(self.test_dataset) > 0:
            if self.load_test_path is not None:
                os.makedirs(os.path.dirname(self.load_test_path), exist_ok=True)
                with open(self.load_test_path, 'w', encoding='utf-8') as file:
                    for data in self.test_dataset:
                        file.write(json.dumps(data) + '\n')
            else:
                os.makedirs(f'../output/{self.model_name}/', exist_ok=True)
                with open(f'../output/{self.model_name}/{self.output_name}_test.jsonl', 'w', encoding='utf-8') as file:
                    for data in self.test_dataset:
                        file.write(json.dumps(data) + '\n')

    def find(self, query, split='train'):
        for data in self.train_dataset if split == 'train' else self.test_dataset:
            if data['query'] == query:
                return data
        return None


class ChessPuzzle(RawPreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.STOCKFISH_PATH = '../model/stockfish_windows/stockfish-windows-x86-64-avx2.exe'
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = datasets.load_dataset('lczerolens/lichess-puzzles')['train'][:100000]
        for i in range(100000):
            if 600 < raw_dataset['Rating'][i] < 1000:
                board = Board(raw_dataset['FEN'][i])
                moves = raw_dataset['Moves'][i].split(' ')
                if len(moves) > 2:
                    board.push(Move.from_uci(moves[0]))
                    self.dataset.append({
                        'FEN': board.fen(),
                        'correct_answer': board.san(Move.from_uci(moves[1]))
                    })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        instruction = "You are a master-level chess player. You will be given a chess position in Forsyth-Edwards Notation (FEN) format. Your task is to analyze the position and choose the best move in Universal Chess Interface (UCI) format for the current player. Note that the move should improve the current player's position by considering both immediate benefits and long-term strategies."
        engine = SimpleEngine.popen_uci(self.STOCKFISH_PATH)
        for data in self.train_dataset:
            query = f"Instruction: {instruction}\nFEN: {data['FEN']}\nOptions: "
            board = Board(data['FEN'])
            info = engine.analyse(board, chess.engine.Limit(time=2.0), multipv=4)
            options = []
            original_correctness = []
            for move_info in info:
                move = move_info["pv"][0]
                score = move_info["score"].relative
                if score.is_mate():
                    if score.mate() > 0:
                        win_chance = 1.0
                    else:
                        win_chance = 0.0
                else:
                    cp = score.score()
                    win_chance = 1 / (1 + 10 ** (-cp / 400))
                options.append(board.san(move))
                original_correctness.append(win_chance)
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                sampled_idxs = random.sample(range(len(options)), len(options))
                choice = ''
                for j in range(len(options)):
                    choice += f"{idx2letter[j]}. {options[sampled_idxs[j]]} "
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query + '\n'
            data['choices'] = choices
            data['correctness'] = correctness
            del data['FEN']
        engine.close()


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

    def process_answer(self, split='train', key=None, peft_dir=None):
        from factscore.factscorer import FactScorer
        fs = FactScorer(
            openai_key=OPENAI_KEY,
            data_dir='../dataset/factscore',
            cache_dir='../.cache/factscore',
        )
        responses_name = 'responses' if key is None else key + '_responses'
        factscore_name = 'factscore' if key is None else key + '_factscore'

        for data in tqdm(self.train_dataset if split == 'train' else self.test_dataset, desc='Generating FactScore'):
            if factscore_name not in data or None in data[factscore_name]:
                try:
                    scores = fs.get_score([data['topic']] * self.response_sample_size, data[responses_name])['score']
                except Exception as e:
                    print(e)
                    scores = [None] * self.response_sample_size
                data[factscore_name] = scores
                self.save_dataset()
        torch.cuda.empty_cache()


class HellaSwag(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = datasets.load_dataset('Rowan/hellaswag')['train']
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


class KnowledgeCrosswords(RawPreferenceDataset):
    """
    Fix correct answer number to 3 only. Easier for future comparison
    """

    def __init__(self, knowledge=True, **kwargs):
        self.knowledge = knowledge
        if self.knowledge:
            self.output_name = f"{kwargs['dataset_name']}_w_knowledge"
        else:
            self.output_name = f"{kwargs['dataset_name']}_wo_knowledge"
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        with open(f'../dataset/{self.dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                if len(data['blanks']) == 3:
                    self.dataset.append({
                        'source': data['source'],
                        'target': data['target'],
                        'relation': data['relation'],
                        'knowledge': data['K'],
                        'options': data['options'],
                        'correct_answer_index': data['answer_all']
                    })

    def precess_dataset(self, sample_size):
        if 0 < sample_size < len(self.dataset):
            random.seed(42)
            self.dataset = random.sample(self.dataset, sample_size)
        for data in self.dataset:
            original_choices, original_correctness = generate_choices_from_candidates(data['options'], data['correct_answer_index'])
            query = f'Instruction: Pick the correct answer for each blank that satisfies all the given constraints.'
            if self.knowledge:
                query += '\nKnowledge: '
                for k in data['knowledge'].values():
                    for term in k:
                        query += f'({term[0]}, {term[1]}, {term[2]}); '
            query += '\nConstraints: '
            for source, relation, target in zip(data['source'], data['relation'], data['target']):
                query += f'({source}, {relation}, {target}); '
            query += '\nOptions: '
            choices = []
            correctness = []
            for i in range(self.response_sample_size):
                choice = ''
                sampled_idxs = random.sample(range(4), 4)
                for j in range(4):
                    choice += f'{idx2letter[j]}. '
                    for k in range(3):
                        choice += f"blank {k + 1}: {original_choices[sampled_idxs[j]][k]}"
                        if k != 2:
                            choice += ', '
                        else:
                            choice += '; '
                choice += '\n\n'
                correctness.append([original_correctness[idx] for idx in sampled_idxs])
                choices.append(choice)
            data['query'] = query
            data['correctness'] = correctness
            data['choices'] = choices
            del data['source']
            del data['relation']
            del data['target']
            del data['knowledge']


class MedMCQA(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x] if letter2idx[x] <= 3 else None
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = datasets.load_dataset('openlifescienceai/medmcqa')
        self.train_dataset = []
        self.test_dataset = []
        for data in raw_dataset['train']:
            self.train_dataset.append({
                'question': data['question'],
                'options': [data['opa'], data['opb'], data['opc'], data['opd']],
                'correct_answer_index':  data['cop']
            })
        for data in raw_dataset['validation']:
            self.test_dataset.append({
                'question': data['question'],
                'options': [data['opa'], data['opb'], data['opc'], data['opd']],
                'correct_answer_index': data['cop']
            })

    def precess_dataset(self, sample_size):
        random.seed(42)
        self.train_dataset = random.sample(self.train_dataset, sample_size // 5 * 4)
        self.test_dataset = random.sample(self.test_dataset, sample_size // 5)
        for data in self.train_dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            original_correctness = [0.0, 0.0, 0.0, 0.0]
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
        for data in self.test_dataset:
            query = f"Question: {data['question']}\nOptions:\n"
            original_correctness = [0.0, 0.0, 0.0, 0.0]
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


class MMLUPro(RawPreferenceDataset):

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
        self.extract_pattern = r'([A-Z])(\.|\. .+)?$'
        self.post_process = lambda x: letter2idx[x]
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro')['test']
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


class NLGraph(RawPreferenceDataset):

    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        super().__init__(**kwargs)
        if kwargs['dataset_name'] == 'NLGraph_shortest_path':
            self.extract_pattern = r'The total weight is (\d+)'
            self.extract_instruction_name = 'shortest_path_extract'
        elif kwargs['dataset_name'] == 'NLGraph_maximum_flow':
            self.extract_pattern = r'The maximum flow is (\d+)'
            self.extract_instruction_name = 'maximum_flow_extract'
        elif kwargs['dataset_name'] == 'NLGraph_matching':
            self.extract_pattern = r'The maximum number of matches is (\d+)'
            self.extract_instruction_name = 'matching_extract'
        else:
            self.extract_pattern = r'The final answer is (\d+)'
            self.extract_instruction_name = 'nlgraph_extract'
        self.post_process = lambda x: int(x)

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


class Science(RawPreferenceDataset):
    def __init__(self, **kwargs):
        self.output_name = kwargs['dataset_name']
        self.extract_pattern = r'The final answer is ([-+]?\d+(?:,\d+)*(?:\.\d+)?(?:[eE][-+]?\d+)?)'
        self.post_process = lambda x: float(x.replace(',', ''))
        super().__init__(**kwargs)

    def load_dataset(self):
        raw_dataset = datasets.load_dataset('xw27/scibench')
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


def load_dataset(dataset_name, model_name, response_sample_size=10, dataset_sample_size=625, load_test_path=None, load_from_exist=True):
    if dataset_name == 'KC' or dataset_name == 'KnowledgeCrosswords':
        dataset = KnowledgeCrosswords(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            knowledge=False,
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name.find('NLGraph') >= 0:
        dataset = NLGraph(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'BioGeneration':
        dataset = BioGeneration(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='default',
            extract_instruction_name='',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'MMLUPro':
        dataset = MMLUPro(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'COM2' or dataset_name == 'CommonSense':
        dataset = COM2(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'HellaSwag':
        dataset = HellaSwag(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'ChessPuzzle':
        dataset = ChessPuzzle(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'MedMCQA':
        dataset = MedMCQA(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='CoT',
            extract_instruction_name='multi_choice_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    elif dataset_name == 'Science':
        dataset = Science(
            dataset_name=dataset_name,
            model_name=model_name,
            instruction_name='default',
            extract_instruction_name='science_extract',
            response_sample_size=response_sample_size,
            dataset_sample_size=dataset_sample_size,
            load_from_exist=load_from_exist,
            load_test_path=load_test_path
        )
    else:
        raise NotImplementedError
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and process answers for construct wrong-over-wrong dataset')
    parser.add_argument('--dataset_name', type=str, default='KC',
                        help='Name of the dataset: KC, NLGraph, NLGraph_shortest_path, NLGraph_maximum_flow, NLGraph_matching, BioGeneration, MMLUPro, COM2, HellaSwag, ChessPuzzle, MedMCQA, Science')
    parser.add_argument('--model_name', type=str, default='gpt-3.5',
                        help='Name of the model: llama-3, gpt-3.5, gpt-4')
    parser.add_argument('--response_sample_size', type=int, default=10, help='Response sample size')
    parser.add_argument('--dataset_sample_size', type=int, default=625, help='Dataset sample size')
    parser.add_argument('--load_from_exist', type=bool, default=True, help='Load from existing dataset or not')
    parser.add_argument('--action', type=str, default='', help='Action flags for what to do with dataset: g, p, gp')

    args = parser.parse_args()

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        response_sample_size=args.response_sample_size,
        dataset_sample_size=args.dataset_sample_size,
        load_from_exist=args.load_from_exist
    )
    if 'g' in args.action:
        dataset.generate_answer()
        dataset.save_dataset()
    if 'p' in args.action:
        dataset.process_answer()
        dataset.save_dataset()
