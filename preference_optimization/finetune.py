import os
import json
import random
import trl
import torch
import argparse
from datetime import datetime
from abc import abstractmethod
from datasets import Dataset, concatenate_datasets
from trl import DPOTrainer
from unsloth import FastLanguageModel

from preference_generation.dataset import load_dataset
from preference_generation.utils import HF_KEY
from preference_optimization.utils import *


class PreferenceDatasetCollector:
    model_name_list: tuple
    output_name: str

    def __init__(self, preference_source, dataset_name):
        self.dataset_name = dataset_name
        self.train_dataset_dict = {
            'prompt': [],
            'chosen': [],
            'rejected': []
        }
        if preference_source == 'all':
            self.model_name_list = ('llama-3', 'gpt-3.5', 'gpt-4')
        else:
            self.model_name_list = ('llama-3',)
        self.filter_train_dataset()

    @abstractmethod
    def filter_train_dataset(self):
        pass

    def get_dataset(self):
        return Dataset.from_dict(self.train_dataset_dict)

    def is_correct(self, idx, data):
        if self.dataset_name.find('NLGraph') >= 0:
            return data['extracted_answers'][idx] == int(data['correct_answer'])
        elif self.dataset_name == 'Science':
            return abs(data['extracted_answers'][idx] - data['correct_answer']) < abs(data['correct_answer']) * 0.05
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][idx] >= 0.9
        else:
            return data['correctness'][idx][data['extracted_answers'][idx]] == max(data['correctness'][idx])

    def is_same_label(self, i, j, data):
        if self.dataset_name.find('NLGraph') >= 0 or self.dataset_name == 'Science':
            return data['extracted_answers'][i] == data['extracted_answers'][j]
        elif self.dataset_name == 'MedMCQA':
            return False
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][i] == data['factscore'][j]
        else:
            return data['correctness'][i][data['extracted_answers'][i]] == data['correctness'][j][
                data['extracted_answers'][j]]

    def get_correctness(self, idx, data):
        if self.dataset_name.find('NLGraph') >= 0:
            return -abs(data['extracted_answers'][idx] - int(data['correct_answer']))
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][idx]
        else:
            return data['correctness'][idx][data['extracted_answers'][idx]]

    def is_valid(self, i, j, data):
        return ('extracted_answers' in data and data['extracted_answers'][i] is not None and
                data['extracted_answers'][j] is not None) or ('factscore' in data and data['factscore'][i] is not None and data['factscore'][j] is not None)


class DirectCompareDatasetCollector(PreferenceDatasetCollector):
    def __init__(self, eval_model_name='gpt-4', filtered=True, **kwargs):
        self.eval_model_name = eval_model_name
        self.filtered = filtered
        self.output_name = ('_filtered_' if filtered else '_') + eval_model_name
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            with open(f'../output/pairwise/{model_name}/{self.eval_model_name}/{dataset_name_translator[self.dataset_name]}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    if not self.filtered:
                        if data['extracted_evaluation'] is not None:
                            if data['extracted_evaluation'] == 1:
                                self.train_dataset_dict['prompt'].append(data['query'])
                                self.train_dataset_dict['chosen'].append(data['response_1'])
                                self.train_dataset_dict['rejected'].append(data['response_2'])
                            else:
                                self.train_dataset_dict['prompt'].append(data['query'])
                                self.train_dataset_dict['chosen'].append(data['response_2'])
                                self.train_dataset_dict['rejected'].append(data['response_1'])
                    else:
                        if data['extracted_evaluation'] is not None and data['reversed_extracted_evaluation'] is not None:
                            if data['extracted_evaluation'] != data['reversed_extracted_evaluation']:
                                if data['extracted_evaluation'] == 1:
                                    self.train_dataset_dict['prompt'].append(data['query'])
                                    self.train_dataset_dict['chosen'].append(data['response_1'])
                                    self.train_dataset_dict['rejected'].append(data['response_2'])
                                else:
                                    self.train_dataset_dict['prompt'].append(data['query'])
                                    self.train_dataset_dict['chosen'].append(data['response_2'])
                                    self.train_dataset_dict['rejected'].append(data['response_1'])


class ScoreCompareDatasetCollector(PreferenceDatasetCollector):

    def __init__(self, eval_model_name='gpt-4', reward_name='reward_5', top_p=0.5, **kwargs):
        self.eval_model_name = eval_model_name
        self.reward_name = self.eval_model_name + '_' + reward_name
        self.top_p = top_p
        self.output_name = '_' + str(top_p) + '_' + eval_model_name
        super().__init__(**kwargs)

    def is_valid(self, i, j, data):
        return data[self.reward_name][i] is not None and data[self.reward_name][j] is not None and \
            (('extracted_answers' in data and data['extracted_answers'][i] is not None and data['extracted_answers'][j] is not None) or
             ('factscore' in data and data['factscore'][i] is not None and data['factscore'][j] is not None))

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            dataset = load_dataset(dataset_name_translator[self.dataset_name], model_name)
            unfiltered_train_dataset = []
            for data in dataset.train_dataset:
                for i in range(len(data['responses'])):
                    for j in range(i + 1, len(data['responses'])):
                        if self.is_valid(i, j, data):
                            if not self.is_correct(i, data) and not self.is_correct(j, data) and not self.is_same_label(i, j, data):
                                if data[self.reward_name][i] > data[self.reward_name][j]:
                                    unfiltered_train_dataset.append({
                                        'prompt': data['query'],
                                        'chosen': data['responses'][i],
                                        'rejected': data['responses'][j],
                                        'gap': data[self.reward_name][i] - data[self.reward_name][j]
                                    })
                                else:
                                    unfiltered_train_dataset.append({
                                        'prompt': data['query'],
                                        'chosen': data['responses'][j],
                                        'rejected': data['responses'][i],
                                        'gap': data[self.reward_name][j] - data[self.reward_name][i]
                                    })
            sorted_dataset = sorted(unfiltered_train_dataset, key=lambda x: x['gap'], reverse=True)
            sorted_dataset = sorted_dataset[: round(len(sorted_dataset) * self.top_p)]
            for data in sorted_dataset:
                self.train_dataset_dict['prompt'].append(data['prompt'])
                self.train_dataset_dict['chosen'].append(data['chosen'])
                self.train_dataset_dict['rejected'].append(data['rejected'])


class OracleDatasetCollector(PreferenceDatasetCollector):
    def __init__(self, preference_filter, **kwargs):
        self.output_name = ''
        self.preference_filter = preference_filter
        super().__init__(**kwargs)

    def is_collected(self, i, j, data):
        if self.preference_filter == 'wow':
            return not self.is_correct(i, data) and not self.is_correct(j, data) and not self.is_same_label(i, j, data)
        else:
            return (self.is_correct(i, data) ^ self.is_correct(j, data)) and not self.is_same_label(i, j, data)

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            dataset = load_dataset(dataset_name_translator[self.dataset_name], model_name)
            for data in dataset.train_dataset:
                for i in range(len(data['responses'])):
                    for j in range(i + 1, len(data['responses'])):
                        if self.is_valid(i, j, data):
                            if self.is_collected(i, j, data):
                                if self.get_correctness(i, data) > self.get_correctness(j, data):
                                    self.train_dataset_dict['prompt'].append(data['query'])
                                    self.train_dataset_dict['chosen'].append(data['responses'][i])
                                    self.train_dataset_dict['rejected'].append(data['responses'][j])
                                else:
                                    self.train_dataset_dict['prompt'].append(data['query'])
                                    self.train_dataset_dict['chosen'].append(data['responses'][j])
                                    self.train_dataset_dict['rejected'].append(data['responses'][i])


def preference_optimization(
        preference_source,
        dataset_name,
        eval_model_name,
        preference_type,
        trainer_name,
        top_p=0.5,
        filtered=True,
        load_from_exist=True,
        learning_rate=5e-5,
        lr_scheduler_type='cosine',
        weight_decay=1e-5,
        num_train_epochs=3,
        beta=0.1,
        warmup_ratio=0.1
):
    if preference_type.find('direct') >= 0:
        dataset_collector = DirectCompareDatasetCollector(
            preference_source=preference_source,
            dataset_name=dataset_name,
            eval_model_name=eval_model_name,
            filtered=filtered
        )
    elif preference_type.find('score') >= 0:
        dataset_collector = ScoreCompareDatasetCollector(
            preference_source=preference_source,
            dataset_name=dataset_name,
            eval_model_name=eval_model_name,
            top_p=top_p
        )
    elif preference_type.find('oracle') >= 0:
        dataset_collector = OracleDatasetCollector(
            preference_source=preference_source,
            preference_filter='wow',
            dataset_name=dataset_name
        )
    else:
        dataset_collector = None
    if preference_type.find('row') >= 0:
        row_dataset_collector = OracleDatasetCollector(
            preference_source=preference_source,
            preference_filter='row',
            dataset_name=dataset_name
        )
    else:
        row_dataset_collector = None

    grid_search_name = f"lr={learning_rate:.1e}_scheduler={lr_scheduler_type}_decay={weight_decay:.1e}_epochs={num_train_epochs}_beta={beta:.1e}_warmup={warmup_ratio:.1e}_timestamp={datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_name = f"{preference_source}_{preference_type}{dataset_collector.output_name if dataset_collector is not None else ''}_{trainer_name}"
    if load_from_exist:
        os.makedirs(f'../output2/{dataset_name}/model/{output_name}/', exist_ok=True)
        parent_path = f'../output2/{dataset_name}/model/{output_name}/'
        grid_search_subdirs = [name for name in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, name))]
        for subdir in grid_search_subdirs:
            if subdir.find(grid_search_name.split('_timestamp=')[0]) >= 0 \
                    and os.path.exists(os.path.join(parent_path, subdir, 'log_all.json')) \
                    and os.path.exists(os.path.join(parent_path, subdir, 'adapter_model.safetensors')):
                return
    random.seed(42)
    if dataset_collector is not None and row_dataset_collector is None:
        dataset = dataset_collector.get_dataset()
        select_idxs = random.sample(range(len(dataset)), min(4000, len(dataset)))
        dataset = dataset.select(select_idxs)
    elif dataset_collector is None and row_dataset_collector is not None:
        dataset = row_dataset_collector.get_dataset()
        select_idxs = random.sample(range(len(dataset)), min(4000, len(dataset)))
        dataset = dataset.select(select_idxs)
    elif dataset_collector is not None and row_dataset_collector is not None:
        dataset = dataset_collector.get_dataset()
        row_dataset = row_dataset_collector.get_dataset()
        select_idxs = random.sample(range(len(dataset)), min(2000, len(dataset), len(row_dataset)))
        dataset = dataset.select(select_idxs)
        select_idxs = random.sample(range(len(row_dataset)), min(2000, len(dataset), len(row_dataset)))
        row_dataset = row_dataset.select(select_idxs)
        dataset = concatenate_datasets([dataset, row_dataset])
        dataset = dataset.shuffle(seed=42)
    else:
        raise NotImplementedError

    dataset = dataset.train_test_split(test_size=0.1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        device_map='auto',
        load_in_4bit=True,
        token=HF_KEY
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    def format_chat_template(row):
        row['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
        row['chosen'] = row['chosen'] + "<|eot_id|>\n"
        row['rejected'] = row['rejected'] + "<|eot_id|>\n"
        return row
    dataset = dataset.map(format_chat_template)

    # TODO: Change training arguments here
    if trl.__version__ == '0.9.6':
        from trl import DPOConfig, CPOTrainer, ORPOTrainer, CPOConfig, ORPOConfig
        args_dict = {
            "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 1,
            # "do_eval": False,
            "do_eval": True,
            "per_device_eval_batch_size": 1,
            "eval_strategy": 'steps',
            "eval_steps": 1000,
            "save_strategy": 'steps',
            "save_steps": 1000,
            "logging_strategy": 'steps',
            "logging_steps": 200,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "output_dir": f'../output2/{dataset_name}/model/{output_name}/{grid_search_name}/',
            "max_length": 1536,
            "max_prompt_length": 512,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "warmup_ratio": warmup_ratio,
        }
        if trainer_name in ('dpo', 'rso', 'ipo', 'sppo'):
            trainer_map = {
                'dpo': 'sigmoid',
                'rso': 'hinge',
                'ipo': 'ipo',
                'sppo': 'sppo_hard'
            }
            dpo_config = DPOConfig(loss_type=trainer_map[trainer_name], beta=beta, **args_dict)
            trainer = DPOTrainer(
                model,
                args=dpo_config,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                tokenizer=tokenizer,
                # model_adapter_name="default",
                # ref_adapter_name="reference",
            )
        elif trainer_name in ('cpo', 'simpo'):
            cpo_config = CPOConfig(**args_dict)
            if trainer_name == 'simpo':
                cpo_config.loss_type = 'simpo'
                cpo_config.cpo_alpha = 0.0
            trainer = CPOTrainer(
                model,
                args=cpo_config,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                tokenizer=tokenizer,
                # model_adapter_name="default",
                # ref_adapter_name="reference",
            )
        elif trainer_name == 'orpo':
            orpo_config = ORPOConfig(**args_dict)
            trainer = ORPOTrainer(
                model,
                args=orpo_config,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                tokenizer=tokenizer,
                # model_adapter_name="default",
                # ref_adapter_name="reference",
            )
        else:
            raise NotImplementedError
    else:  # trl == 0.8.6
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            # do_eval=False,
            do_eval=True,
            per_device_eval_batch_size=1,
            eval_strategy='steps',
            eval_steps=1000,
            save_strategy='steps',
            save_steps=1000,
            logging_strategy='steps',
            logging_steps=200,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=f'../output2/{dataset_name}/model/{output_name}/{grid_search_name}/',
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
        )
        if trainer_name == 'dpo':
            trainer = DPOTrainer(
                model,
                beta=beta,
                args=training_args,
                max_length=1536,
                max_prompt_length=512,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError
    trainer.train()
    trainer.save_model()

    # Save anything you are interested about loss here.
    train_losses = []
    val_losses = []
    for log in trainer.state.log_history:
        if 'loss' in log:
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            val_losses.append(log['eval_loss'])
    out_json = {
        'loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': min(val_losses) if len(val_losses) > 0 else 1.0
    }
    with open(f'../output2/{dataset_name}/model/{output_name}/{grid_search_name}/log_all.json', 'w', encoding='utf-8') as file:
        json.dump(out_json, file)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preference Optimization Script")
    parser.add_argument('--preference_source', type=str, default='all',
                        help='Source where preferences are collected: all, self')
    parser.add_argument('--dataset_name', type=str, default='KnowledgeCrosswords',
                        help='Name of the dataset: KnowledgeCrosswords, BioGeneration, CommonSense, NLGraph_SP, MedMCQA, Science')
    parser.add_argument('--eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model: gpt-4')
    parser.add_argument('--preference_type', type=str, default='oracle',
                        help='Type of preference: oracle, direct, score, row, row_oracle, row_direct, row_score.')
    parser.add_argument('--trainer_name', type=str, default='dpo',
                        help='Name of the trainer: dpo, rso, ipo, sppo, cpo, simpo, orpo')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value: 0.5, 0.1')
    parser.add_argument('--filtered', type=bool, default=True,
                        help='Boolean flag to indicate if filtering is applied: True')
    parser.add_argument('--load_from_exist', type=bool, default=True)
    args = parser.parse_args()

    preference_optimization(**vars(args))
