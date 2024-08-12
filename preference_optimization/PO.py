import os
import json
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


class PreferenceDatasetCollector:
    model_name_list: tuple = ('llama-3', 'gpt-3.5', 'gpt-4')
    output_name: str

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.train_dataset_dict = {
            'prompt': [],
            'chosen': [],
            'rejected': []
        }
        self.filter_train_dataset()

    @abstractmethod
    def filter_train_dataset(self):
        pass

    def get_dataset(self):
        return Dataset.from_dict(self.train_dataset_dict)


class DirectCompareDatasetCollector(PreferenceDatasetCollector):
    def __init__(self, eval_model_name='gpt-4', filtered=True, **kwargs):
        self.eval_model_name = eval_model_name
        self.filtered = filtered
        self.output_name = eval_model_name + '_direct' + ('_filtered' if filtered else '')
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        for model_name in self.model_name_list:
            if not os.path.exists(f'../output/pairwise/{model_name}/{self.eval_model_name}/{self.dataset_name}.jsonl'):
                raise FileNotFoundError(
                    f'../output/pairwise/{model_name}/{self.eval_model_name}/{self.dataset_name}.jsonl')
            with open(f'../output/pairwise/{model_name}/{self.eval_model_name}/{self.dataset_name}.jsonl', 'r',
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
        self.output_name = eval_model_name + '_score_' + str(top_p)
        super().__init__(**kwargs)

    def filter_train_dataset(self):
        def is_correct(idx, data):
            if self.dataset_name.find('NLGraph') >= 0:
                return data['extracted answers'][idx] == int(data['correct_answer'])
            elif self.dataset_name == 'BioGeneration':
                return data['factscore'][idx] == 1.0
            else:
                return data['correctness'][idx][data['extracted answers'][idx]] == max(data['correctness'][idx])

        def is_same_label(i, j, data):
            if self.dataset_name.find('NLGraph') >= 0:
                return data['extracted answers'][i] == data['extracted answers'][j]
            elif self.dataset_name == 'BioGeneration':
                return data['factscore'][i] == data['factscore'][j]
            else:
                return data['correctness'][i][data['extracted answers'][i]] == data['correctness'][j][
                    data['extracted answers'][j]]

        reward_name = self.eval_model_name + '_' + self.reward_name
        for model_name in self.model_name_list:
            dataset = load_dataset(self.dataset_name, model_name)
            unfiltered_train_dataset = []
            for data in dataset.train_dataset:
                for i in range(len(data['responses'])):
                    for j in range(i + 1, len(data['responses'])):
                        if data[reward_name][i] is not None and data[reward_name][j] is not None and \
                                data['extracted answers'][i] is not None and data['extracted answers'][j] is not None:
                            if not is_correct(i, data) and not is_correct(j, data) and not is_same_label(i, j, data):
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


def preference_optimization(
        dataset_name,
        eval_model_name,
        preference_type,
        trainer_name,
        top_p=0.5,
        filtered=True,
):
    if preference_type == 'direct':
        dataset_collector = DirectCompareDatasetCollector(eval_model_name=eval_model_name, dataset_name=dataset_name, filtered=filtered)
    elif preference_type == 'score':
        dataset_collector = ScoreCompareDatasetCollector(eval_model_name=eval_model_name, dataset_name=dataset_name, top_p=top_p)
    else:
        raise NotImplementedError

    dataset = dataset_collector.get_dataset()
    dataset = dataset.shuffle(42)
    dataset = dataset.train_test_split(test_size=0.1)

    # login(token='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')
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     llm_int8_threshold=6.0,
    #     llm_int8_has_fp16_weight=False,
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
    # # TODO: do I need to?
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
    # tokenizer.padding_side = 'left'
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

    # TODO: What's the correct way to apply chat template?
    def format_chat_template(row):
        row['prompt'] = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        row['chosen'] = row['chosen'] + "<|eot_id|>\n"
        row['rejected'] = row['rejected'] + "<|eot_id|>\n"
        return row
    dataset = dataset.map(format_chat_template)

    # TODO: Change training arguments here
    training_args = TrainingArguments(
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        do_eval=False,
        # do_eval=True,
        # per_device_eval_batch_size=4,
        # eval_strategy='epoch',
        # save_strategy='epoch',
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # save_total_limit=1,
        # load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=f'../output/model/{dataset_name}/{eval_model_name}_{dataset_collector.output_name}_{trainer_name}/',
    )
    if trainer_name == 'dpo':
        trainer = DPOTrainer(
            model,
            args=training_args,
            beta=0.1,
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
    trainer.save_model(f'../output/model/{dataset_name}/{dataset_collector.output_name}_{trainer_name}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preference Optimization Script")

    parser.add_argument('dataset_name', type=str, default='KC', help='Name of the dataset')
    parser.add_argument('eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model')
    parser.add_argument('preference_type', type=str, default='direct', help='Type of preference (e.g., "direct")')
    parser.add_argument('trainer_name', type=str, default='dpo', help='Name of the trainer (e.g., "dpo")')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value (default: 0.5)')
    parser.add_argument('--filtered', type=bool, default=True, help='Boolean flag to indicate if filtering is applied (default: True)')

    args = parser.parse_args()

    preference_optimization(
        args.dataset_name,
        args.eval_model_name,
        args.preference_type,
        args.trainer_name,
        top_p=args.top_p,
        filtered=args.filtered,
    )
    preference_optimization('KC', 'gpt-4', 'direct', 'dpo')
