import numpy as np
import os
import json
import re
import torch
import random
import argparse
from tqdm import tqdm
from math import floor
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from preference_generation.dataset import load_dataset
from preference_generation.utils import batch_query_openai, batch_query_open_sourced_llm, HF_KEY


def calculate_accuracy_score(predictions, labels, is_corrects=None, top_p=1.0, detailed=None):
    """
    :param is_corrects: list[list[bool: mask for each answer being correct,
    correct answers will not count in; when not given, all answers will be considered incorrect]]
    :param detailed: whether show detailed accuracy, if provided, it should be the number of bins
    :param predictions: list[list[float: prediction scores]]
    :param labels: list[list[float: correctness in [0.0, 1.0]]]
    :param top_p: margin parameter
    :return:
    """
    assert isinstance(top_p, float)

    if is_corrects is None:
        is_corrects = []
        for prediction in predictions:
            is_corrects.append([False] * len(prediction))

    gaps = []
    corrects = []
    for prediction, label, is_correct in zip(predictions, labels, is_corrects):
        for i in range(len(prediction)):
            for j in range(i + 1, len(prediction)):
                if not is_correct[i] and not is_correct[j] and label[i] != label[j]:
                    gaps.append(abs(prediction[i] - prediction[j]))
                    if (prediction[i] - prediction[j]) * (label[i] - label[j]) > 0:
                        corrects.append((1, label[i], label[j], prediction[i], prediction[j]))
                    else:
                        corrects.append((0, label[i], label[j], prediction[i], prediction[j]))

    gaps = np.array(gaps)
    corrects = np.array(corrects)
    num_top_p = int(len(gaps) * top_p)
    top_p_indices = np.argsort(gaps)[-num_top_p:]
    print(f'Overall accuracy is {np.array([c for c, _, _, _, _ in corrects[top_p_indices]]).mean(): .3f}')

    if detailed is not None:
        assert isinstance(detailed, int)
        absolutes = []
        for label, is_correct in zip(labels, is_corrects):
            for l, c in zip(label, is_correct):
                if not c:
                    absolutes.append(l)
        bin_size = (max(absolutes) - min(absolutes)) / detailed
        bin_start = min(absolutes)
        for i in range(detailed):
            print(f"Bin {i}'s range is [{bin_start + i * bin_size: .3f}, {bin_start + (i + 1) * bin_size: .3f})")
        detailed_accurate_count = [[0 for _ in range(detailed)] for __ in range(detailed)]
        detailed_all_count = [[0 for _ in range(detailed)] for __ in range(detailed)]
        for c, label_i, label_j, prediction_i, prediction_j in corrects[top_p_indices]:
            i_bin_index = min(floor((label_i - bin_start) / bin_size), detailed - 1)
            j_bin_index = min(floor((label_j - bin_start) / bin_size), detailed - 1)
            detailed_all_count[i_bin_index][j_bin_index] += 1
            if (prediction_i - prediction_j) * (label_i - label_j) > 0:
                detailed_accurate_count[i_bin_index][j_bin_index] += 1
        for i in range(detailed):
            for j in range(i + 1, detailed):
                print(f'Accuracy for {j} over {i} is: {(detailed_accurate_count[i][j] + detailed_accurate_count[j][i]) / (detailed_all_count[i][j] + detailed_all_count[j][i] + 1e-8): .3f}')


def calculate_task_accuracy_and_wow_number(dataset_name, model_name):
    dataset = load_dataset(dataset_name, model_name)
    is_corrects = []
    confidence = []
    for data in dataset.train_dataset:
        confidence += [np.exp(-log_prob) for log_prob in data['log_probs']]
        if dataset_name.find('NLGraph') >= 0 or dataset_name == 'Science':
            is_corrects += [int(data['correct_answer']) == e for e in data['extracted_answers'] if e is not None]
        elif dataset_name == 'BioGeneration':
            is_corrects += [fs for fs in data['factscore'] if fs is not None]
        else:
            correct_answer = max(data['correctness'][0])
            is_corrects += [c[e] == correct_answer for e, c in zip(data['extracted_answers'], data['correctness']) if e is not None]
    print(f'Task accuracy is {sum(is_corrects) / len(dataset.train_dataset) / dataset.response_sample_size}')
    print(f'Task confidence is {sum(confidence) / len(dataset.train_dataset) / dataset.response_sample_size}')

    if os.path.exists(f'../output/pairwise/{model_name}/gpt-4/{dataset_name}.jsonl'):
        with open(f'../output/pairwise/{model_name}/gpt-4/{dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            data = file.readlines()
            print(f'Number of wrong over wrong pairs is {len(data)}')
    else:
        print('Please run the pairwise evaluation before counting overall WoW pairs.')


def calculate_accuracy_from_response(dataset_name, model_name):
    log_probs = []
    consistencies = []
    lengths = []
    labels = []
    is_corrects = []
    dataset = load_dataset(dataset_name, model_name)

    login(token=HF_KEY)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

    if dataset_name.find('NLGraph') >= 0 or dataset_name == 'Science':
        for data in dataset.train_dataset:
            log_probs.append([-l for e, l in zip(data['extracted_answers'], data['log_probs']) if e is not None])
            consistency = [e for e in data['extracted_answers'] if e is not None]
            counter = Counter(consistency)
            consistencies.append([counter[c] / len(consistency) for c in consistency])
            lengths.append([len(tokenizer(r)['input_ids']) for e, r in zip(data['extracted_answers'], data['responses']) if e is not None])
            labels.append([-abs(int(data['correct_answer']) - e) for e in data['extracted_answers'] if e is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        for data in dataset.train_dataset:
            log_probs.append([-l for fs, l in zip(data['factscore'], data['log_probs']) if fs is not None])
            consistencies.append([-l for fs, l in zip(data['factscore'], data['log_probs']) if fs is not None])
            lengths.append([len(tokenizer(r)['input_ids']) for fs, r in zip(data['factscore'], data['responses']) if fs is not None])
            labels.append([fs for fs in data['factscore'] if fs is not None])
            is_corrects.append([fs >= 0.9 for fs in data['factscore'] if fs is not None])
    else:
        for data in dataset.train_dataset:
            log_probs.append([-l for e, l in zip(data['extracted_answers'], data['log_probs']) if e is not None])
            consistency = [e for e in data['extracted_answers'] if e is not None]
            counter = Counter(consistency)
            consistencies.append([counter[c] / len(consistency) for c in consistency])
            lengths.append([len(tokenizer(r)['input_ids']) for e, r in zip(data['extracted_answers'], data['responses']) if e is not None])
            labels.append([c[e] for e, c in zip(data['extracted_answers'], data['correctness']) if e is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(lengths, labels, is_corrects=is_corrects, top_p=1.0)
    calculate_accuracy_score(lengths, labels, is_corrects=is_corrects, top_p=0.5)
    calculate_accuracy_score(lengths, labels, is_corrects=is_corrects, top_p=0.1)
    calculate_accuracy_score(consistencies, labels, is_corrects=is_corrects, top_p=1.0)
    calculate_accuracy_score(consistencies, labels, is_corrects=is_corrects, top_p=0.5)
    calculate_accuracy_score(consistencies, labels, is_corrects=is_corrects, top_p=0.1)
    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=1.0)
    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=0.5)
    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=0.1)
    del tokenizer


def calculate_accuracy_ask_llm_pairwise(
        dataset_name,
        model_name,
        evaluate_instruction_name,
        evaluate_model_name,
        filtered=False,
        load_from_exist=False
):
    if load_from_exist and os.path.exists(f'../output/pairwise/{model_name}/{evaluate_model_name}/{dataset_name}.jsonl'):
        evaluation_jsonl = []
        with open(f'../output/pairwise/{model_name}/{evaluate_model_name}/{dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                evaluation_jsonl.append(json.loads(line.strip()))
    else:
        dataset = load_dataset(dataset_name, model_name)
        with open(f'../instruction/{evaluate_instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        prompt_list = []
        evaluation_jsonl = []
        for data in dataset.train_dataset:
            if dataset_name.find('NLGraph') >= 0 or dataset_name == 'Science':
                responses = [r for r, e in zip(data['responses'], data['extracted_answers']) if e is not None]
                label = [-abs(int(data['correct_answer']) - e) for e in data['extracted_answers'] if e is not None]
                is_correct = [l == 0 for l in label]
            elif dataset_name == 'BioGeneration':
                responses = [r for r, fs in zip(data['responses'], data['factscore']) if fs is not None]
                label = [fs for fs in data['factscore'] if fs is not None]
                is_correct = [fs >= 0.9 for fs in data['factscore'] if fs is not None]
            else:
                responses = [r for r, e in zip(data['responses'], data['extracted_answers']) if e is not None]
                label = [c[e] for e, c in zip(data['extracted_answers'], data['correctness']) if e is not None]
                is_correct = [l == max(data['correctness'][0]) for l in label]
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    if not is_correct[i] and not is_correct[j] and label[i] != label[j]:
                        if 'choices' in data:
                            query = data['query'][:-9]
                        else:
                            query = data['query']
                        evaluation_jsonl.append({
                            'query': query,
                            'response_1': responses[i],
                            'response_2': responses[j],
                            'ground_truth': 1 if label[i] > label[j] else 2
                        })
                        prompt_list.append([{
                            'role': 'user',
                            'content': instruction.format(query, responses[i], responses[j])
                        }])
        if evaluate_model_name == 'gpt-4' and len(evaluation_jsonl) > 10000:
            random.seed(42)
            sampled_idxs = random.sample(range(len(evaluation_jsonl)), 10000)
            sampled_idxs = sorted(sampled_idxs)
            evaluation_jsonl = [evaluation_jsonl[idx] for idx in sampled_idxs]
            prompt_list = [prompt_list[idx] for idx in sampled_idxs]
        if evaluate_model_name == 'gpt-4':
            lt_pairs, evaluations = batch_query_openai(prompt_list, model_name='gpt-4o', mode='evaluate')
        elif evaluate_model_name == 'gpt-3.5':
            lt_pairs, evaluations = batch_query_openai(prompt_list, model_name='gpt-3.5-turbo', mode='evaluate')
        elif evaluate_model_name == 'llama-3':
            lt_pairs, evaluations = batch_query_open_sourced_llm(prompt_list, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='evaluate_256')
        else:
            raise NotImplementedError
        pattern = re.compile(r'Preferred output: (\d+)')
        for evaluation, eval_json in zip(evaluations, evaluation_jsonl):
            eval_json['evaluation'] = evaluation
            match = re.search(pattern, evaluation)
            if match is not None:
                eval_json['extracted_evaluation'] = int(match.group(1))
            else:
                eval_json['extracted_evaluation'] = None

        os.makedirs(f'../output/pairwise/{model_name}/{evaluate_model_name}', exist_ok=True)
        with open(f'../output/pairwise/{model_name}/{evaluate_model_name}/{dataset_name}.jsonl', 'w', encoding='utf-8') as file:
            for data in evaluation_jsonl:
                file.write(json.dumps(data) + '\n')
    if filtered and not (load_from_exist and 'reversed_evaluation' in evaluation_jsonl[0]):
        with open(f'../instruction/{evaluate_instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        prompt_list = []
        for eval_json in evaluation_jsonl:
            prompt_list.append([{
                'role': 'user',
                'content': instruction.format(eval_json['query'], eval_json['response_2'], eval_json['response_1'])
            }])
        if evaluate_model_name == 'gpt-4':
            _, evaluations = batch_query_openai(prompt_list, model_name='gpt-4o', mode='evaluate')
        elif evaluate_model_name == 'gpt-3.5':
            _, evaluations = batch_query_openai(prompt_list, model_name='gpt-3.5-turbo', mode='evaluate')
        elif evaluate_model_name == 'llama-3':
            _, evaluations = batch_query_open_sourced_llm(prompt_list, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='evaluate_256')
        else:
            raise NotImplementedError
        pattern = re.compile(r'Preferred output: (\d+)')
        for evaluation, eval_json in zip(evaluations, evaluation_jsonl):
            eval_json['reversed_evaluation'] = evaluation
            match = re.search(pattern, evaluation)
            if match is not None:
                eval_json['reversed_extracted_evaluation'] = int(match.group(1))
            else:
                eval_json['reversed_extracted_evaluation'] = None

        os.makedirs(f'../output/pairwise/{model_name}/{evaluate_model_name}', exist_ok=True)
        with open(f'../output/pairwise/{model_name}/{evaluate_model_name}/{dataset_name}.jsonl', 'w', encoding='utf-8') as file:
            for data in evaluation_jsonl:
                file.write(json.dumps(data) + '\n')

    accurate_count = 0
    all_count = 0
    for evaluation in evaluation_jsonl:
        if not filtered:
            all_count += 1
            if evaluation['extracted_evaluation'] is not None and evaluation['extracted_evaluation'] == evaluation['ground_truth']:
                accurate_count += 1
        else:
            if evaluation['extracted_evaluation'] is not None and evaluation['reversed_extracted_evaluation'] is not None and evaluation['extracted_evaluation'] != evaluation['reversed_extracted_evaluation']:
                all_count += 1
                if evaluation['extracted_evaluation'] == evaluation['ground_truth']:
                    accurate_count += 1
    print(f'Overall accuracy is {accurate_count / (all_count + 1e-8): .3f}')
    if filtered:
        print(f'Number of wrong over wrong pairs is: {all_count}')


def calculate_accuracy_ask_llm_score(
        dataset_name,
        model_name,
        reward_name,
        evaluate_model_name,
        load_from_exist=False,
):
    reward_name = evaluate_model_name + '_' + reward_name
    dataset = load_dataset(dataset_name, model_name)
    if not (load_from_exist and reward_name in dataset.train_dataset[0]):
        _, _, batch_size = reward_name.split('_')
        with open(f'../instruction/evaluate_reward_{batch_size}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        batch_size = int(batch_size)
        prompt_list = []
        for data in dataset.train_dataset:
            if 'choices' in data:
                query = data['query'][:-9]
            else:
                query = data['query']
            for i in range(0, len(data['responses']), batch_size):
                args = [query]
                for j in range(i, i + batch_size):
                    args.append(data['responses'][j])
                prompt_list.append([{
                    'role': 'user',
                    'content': instruction.format(*args)
                }])
        if evaluate_model_name == 'gpt-4':
            _, evaluations = batch_query_openai(prompt_list, model_name='gpt-4o', mode='evaluate')
        elif evaluate_model_name == 'gpt-3.5':
            _, evaluations = batch_query_openai(prompt_list, model_name='gpt-3.5-turbo', mode='evaluate')
        elif evaluate_model_name == 'llama-3':
            _, evaluations = batch_query_open_sourced_llm(prompt_list, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='evaluate_1024')
        else:
            raise NotImplementedError
        pattern = re.compile(r"Score: (\d+)")
        step = 10 // batch_size
        for idx, data in enumerate(dataset.train_dataset):
            rewards = []
            for i in range(step):
                scores = re.findall(pattern, evaluations[idx * step + i])
                scores = list(map(int, scores))
                if len(scores) != batch_size:
                    scores = [None] * batch_size
                rewards += scores
            data[reward_name] = rewards
            data[reward_name + '_responses'] = evaluations[idx * step: idx * step + step]

        dataset.save_dataset()

    rewards = []
    labels = []
    is_corrects = []

    if dataset_name.find('NLGraph') >= 0 or dataset_name == 'Science':
        for data in dataset.train_dataset:
            rewards.append([r for e, r in zip(data['extracted_answers'], data[reward_name]) if e is not None and r is not None])
            labels.append([-abs(int(data['correct_answer']) - e) for e, r in zip(data['extracted_answers'], data[reward_name]) if e is not None and r is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        for data in dataset.train_dataset:
            rewards.append([r for fs, r in zip(data['factscore'], data[reward_name]) if fs is not None and r is not None])
            labels.append([fs for fs in data['factscore'] if fs is not None])
            is_corrects.append([fs >= 0.9 for fs in data['factscore'] if fs is not None])
    else:
        for data in dataset.train_dataset:
            rewards.append([r for e, r in zip(data['extracted_answers'], data[reward_name]) if e is not None and r is not None])
            labels.append([c[e] for e, c, r in zip(data['extracted_answers'], data['correctness'], data[reward_name]) if e is not None and r is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(rewards, labels, is_corrects=is_corrects, top_p=1.0)
    calculate_accuracy_score(rewards, labels, is_corrects=is_corrects, top_p=0.5)
    calculate_accuracy_score(rewards, labels, is_corrects=is_corrects, top_p=0.1)


def calculate_nll(dataset_name, model_name, eval_model_name='llama-3', load_from_exist=True):
    dataset = load_dataset(dataset_name, model_name)
    if not (load_from_exist and eval_model_name + '_log_probs' in dataset.train_dataset[0]):
        if eval_model_name == 'llama-3':
            login(token=HF_KEY)
            model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Meta-Llama-3-8B-Instruct',
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise NotImplementedError
        for data in tqdm(dataset.train_dataset):
            nll_list = []
            for response in data['responses']:
                response_tokens = tokenizer([response], padding=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**response_tokens, labels=response_tokens['input_ids'])
                    nll_list.append(outputs.loss.cpu().item())
            data[eval_model_name + '_log_probs'] = nll_list
        dataset.save_dataset()
    log_probs = []
    labels = []
    is_corrects = []

    if dataset_name.find('NLGraph') >= 0 or dataset_name == 'Science':
        for data in dataset.train_dataset:
            log_probs.append([-l for e, l in zip(data['extracted_answers'], data[eval_model_name + '_log_probs']) if e is not None])
            labels.append([-abs(int(data['correct_answer']) - e) for e in data['extracted_answers'] if e is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        for data in dataset.train_dataset:
            log_probs.append([-l for fs, l in zip(data['factscore'], data[eval_model_name + '_log_probs']) if fs is not None])
            labels.append([fs for fs in data['factscore'] if fs is not None])
            is_corrects.append([fs >= 0.9 for fs in data['factscore'] if fs is not None])
    else:
        for data in dataset.train_dataset:
            log_probs.append([-l for e, l in zip(data['extracted_answers'], data[eval_model_name + '_log_probs']) if e is not None])
            labels.append([c[e] for e, c in zip(data['extracted_answers'], data['correctness']) if e is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=1.0)
    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=0.5)
    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate wrong-over-wrong preference accuracy.')
    parser.add_argument('--dataset_name', type=str, default='KC',
                        help='Name of the dataset: KC, NLGraph, NLGraph_shortest_path, NLGraph_maximum_flow, NLGraph_matching, BioGeneration, MMLUPro, COM2, HellaSwag, ChessPuzzle, MedMCQA, Science')
    parser.add_argument('--model_name', type=str, default='gpt-3.5',
                        help='Name of the generator: llama-3, gpt-3.5, gpt-4')
    parser.add_argument('--eval_model_name', type=str, default='gpt-3.5',
                        help='Name of the evaluator: llama-3, gpt-3.5, gpt-4')
    parser.add_argument('--strategy', type=str, default='self', help='Strategy to elicit wrong-over-wrong preference: self, pair, score')
    parser.add_argument('--load_from_exist', type=bool, default=True, help='Load from existing dataset or not')

    args = parser.parse_args()

    if args.strategy == 'self':
        calculate_accuracy_from_response(args.dataset_name, args.model_name)
    elif args.strategy == 'pair':
        calculate_accuracy_ask_llm_pairwise(args.dataset_name, args.model_name, 'evaluate_pairwise', args.eval_model_name, True, args.load_from_exist)
        calculate_accuracy_ask_llm_pairwise(args.dataset_name, args.model_name, 'evaluate_pairwise', args.eval_model_name, False, args.load_from_exist)
    elif args.strategy == 'score':
        calculate_accuracy_ask_llm_score(args.dataset_name, args.model_name, 'reward_5', args.eval_model_name, args.load_from_exist)
