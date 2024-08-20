import os
import json
import random

import torch
import argparse
from abc import abstractmethod
from datasets import Dataset
from trl import DPOTrainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from unsloth import FastLanguageModel

from preference_generation.metric import load_dataset
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
            return data['extracted answers'][idx] == int(data['correct_answer'])
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][idx] >= 0.9
        else:
            return data['correctness'][idx][data['extracted answers'][idx]] == max(data['correctness'][idx])

    def is_same_label(self, i, j, data):
        if self.dataset_name.find('NLGraph') >= 0:
            return data['extracted answers'][i] == data['extracted answers'][j]
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][i] == data['factscore'][j]
        else:
            return data['correctness'][i][data['extracted answers'][i]] == data['correctness'][j][
                data['extracted answers'][j]]

    def get_correctness(self, idx, data):
        if self.dataset_name.find('NLGraph') >= 0:
            return -abs(data['extracted answers'][idx] - int(data['correct_answer']))
        elif self.dataset_name == 'BioGeneration':
            return data['factscore'][idx]
        else:
            return data['correctness'][idx][data['extracted answers'][idx]]


class DirectCompareDatasetCollector(PreferenceDatasetCollector):
    def __init__(self, eval_model_name='gpt-4', filtered=True, **kwargs):
        self.eval_model_name = eval_model_name
        self.filtered = filtered
        self.output_name = 'direct' + ('_filtered_' if filtered else '_') + eval_model_name
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            if not os.path.exists(f'../output/pairwise/{model_name}/{self.eval_model_name}/{dataset_name_translator[self.dataset_name]}.jsonl'):
                raise FileNotFoundError(
                    f'../output/pairwise/{model_name}/{self.eval_model_name}/{dataset_name_translator[self.dataset_name]}.jsonl')
            with open(f'../output/pairwise/{model_name}/{self.eval_model_name}/{dataset_name_translator[self.dataset_name]}.jsonl', 'r',
                      encoding='utf-8') as file:
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
        self.reward_name = reward_name
        self.top_p = top_p
        self.output_name = 'score_' + str(top_p) + '_' + eval_model_name
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        reward_name = self.eval_model_name + '_' + self.reward_name
        for model_name in self.model_name_list:
            dataset = load_dataset(dataset_name_translator[self.dataset_name], model_name)
            unfiltered_train_dataset = []
            for data in dataset.train_dataset:
                for i in range(len(data['responses'])):
                    for j in range(i + 1, len(data['responses'])):
                        if data[reward_name][i] is not None and data[reward_name][j] is not None and \
                                (('extracted answers' in data and data['extracted answers'][i] is not None and data['extracted answers'][j] is not None) or
                                 ('factscore' in data and data['factscore'][i] is not None and data['factscore'][j] is not None)):
                            if not self.is_correct(i, data) and not self.is_correct(j, data) and not self.is_same_label(i, j, data):
                                if data[reward_name][i] > data[reward_name][j]:
                                    unfiltered_train_dataset.append({
                                        'prompt': data['query'],
                                        'chosen': data['responses'][i],
                                        'rejected': data['responses'][j],
                                        'gap': data[reward_name][i] - data[reward_name][j]
                                    })
                                else:
                                    unfiltered_train_dataset.append({
                                        'prompt': data['query'],
                                        'chosen': data['responses'][j],
                                        'rejected': data['responses'][i],
                                        'gap': data[reward_name][j] - data[reward_name][i]
                                    })
            sorted_dataset = sorted(unfiltered_train_dataset, key=lambda x: x['gap'], reverse=True)
            sorted_dataset = sorted_dataset[: round(len(sorted_dataset) * self.top_p)]
            for data in sorted_dataset:
                self.train_dataset_dict['prompt'].append(data['prompt'])
                self.train_dataset_dict['chosen'].append(data['chosen'])
                self.train_dataset_dict['rejected'].append(data['rejected'])


class OracleDatasetCollector(PreferenceDatasetCollector):
    def __init__(self, **kwargs):
        self.output_name = 'oracle'
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            dataset = load_dataset(dataset_name_translator[self.dataset_name], model_name)
            for data in dataset.train_dataset:
                for i in range(len(data['responses'])):
                    for j in range(i + 1, len(data['responses'])):
                        if ('extracted answers' in data and data['extracted answers'][i] is not None and data['extracted answers'][j] is not None) or \
                                ('factscore' in data and data['factscore'][i] is not None and data['factscore'][j] is not None):
                            if not self.is_correct(i, data) and not self.is_correct(j, data) and not self.is_same_label(i, j, data):
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
        load_from_exist=True
):
    if preference_type == 'direct':
        dataset_collector = DirectCompareDatasetCollector(
            preference_source=preference_source,
            eval_model_name=eval_model_name,
            dataset_name=dataset_name,
            filtered=filtered
        )
    elif preference_type == 'score':
        dataset_collector = ScoreCompareDatasetCollector(
            preference_source=preference_source,
            eval_model_name=eval_model_name,
            dataset_name=dataset_name,
            top_p=top_p
        )
    elif preference_type == 'oracle':
        dataset_collector = OracleDatasetCollector(
            preference_source=preference_source,
            dataset_name=dataset_name
        )
    else:
        raise NotImplementedError
    if load_from_exist and os.path.exists(f'../output2/{dataset_name}/model/{preference_source}_{dataset_collector.output_name}_{trainer_name}/adapter_model.safetensors'):
        return
    dataset = dataset_collector.get_dataset()
    random.seed(42)
    select_idxs = random.sample(range(len(dataset)), min(4000, len(dataset)))
    dataset = dataset.select(select_idxs)
    dataset = dataset.train_test_split(test_size=0.1)

    # login(token='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')
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     'meta-llama/Meta-Llama-3-8B-Instruct',
    #     device_map='auto',
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=bnb_config,
    #     attn_implementation="flash_attention_2",
    #     use_cache=False
    # )
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0,
    #     target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, peft_config=peft_config)
    # model.add_adapter(peft_config=peft_config, adapter_name="reference")
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'

    # TODO: if you want to use unsloth. However, unsloth doesn't support multi-gpus now.
    # TODO: make sure you comment out model_adapter_name and ref_adapter_name
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='meta-llama/Meta-Llama-3-8B-Instruct',
        device_map='auto',
        load_in_4bit=True,
        token='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'
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
        row['prompt'] = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        row['chosen'] = row['chosen'] + "<|eot_id|>\n"
        row['rejected'] = row['rejected'] + "<|eot_id|>\n"
        return row
    dataset = dataset.map(format_chat_template)

    # TODO: Change training arguments here
    training_args = TrainingArguments(
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        # do_eval=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        eval_strategy='steps',
        eval_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=f'../output2/{dataset_name}/model/{preference_source}_{dataset_collector.output_name}_{trainer_name}/',
    )
    if trainer_name == 'dpo':
        trainer = DPOTrainer(
            model,
            args=training_args,
            max_length=1536,
            max_prompt_length=512,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
            # model_adapter_name="default",
            # ref_adapter_name="reference",
        )
    else:
        raise NotImplementedError
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preference Optimization Script")
    parser.add_argument('--preference_source', type=str, default='all', help='Source where preferences are collected: all, self')
    parser.add_argument('--dataset_name', type=str, default='KnowledgeCrosswords',
                        help='Name of the dataset: KnowledgeCrosswords, BioGeneration, CommonSense, NLGraph_SP')
    parser.add_argument('--eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model: gpt-4')
    parser.add_argument('--preference_type', type=str, default='oracle', help='Type of preference: oracle, direct, score.')
    parser.add_argument('--trainer_name', type=str, default='dpo', help='Name of the trainer: dpo')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value: 0.5, 0.1')
    parser.add_argument('--filtered', type=bool, default=True, help='Boolean flag to indicate if filtering is applied: True')
    parser.add_argument('--load_from_exist', type=bool, default=True)

    args = parser.parse_args()

    preference_optimization(
        preference_source=args.preference_source,
        dataset_name=args.dataset_name,
        preference_type=args.preference_type,
        eval_model_name=args.eval_model_name,
        trainer_name=args.trainer_name,
        top_p=args.top_p,
        filtered=args.filtered,
        load_from_exist=args.load_from_exist
    )
