import json
import os
from datetime import datetime
from heapq import nlargest

from preference_generation.metric import load_dataset
from preference_optimization.utils import *

import argparse
import numpy as np


def generate_evaluation_responses(dataset, peft_dir):
    # generate responses
    dataset.generate_answer('CoT', split='test', peft_dir=peft_dir)
    dataset.save_dataset()
    if dataset.dataset_name != 'BioGeneration':  # TODO: mask this line when testing on BioGeneration
        # process responses
        eval_instruction_name, pattern = get_extract_instruction_name_and_pattern(
            dataset_name_translator[dataset.dataset_name]
        )
        dataset.extract_pattern = pattern
        dataset.process_answer('CoT', eval_instruction_name, split='test', peft_dir=peft_dir)
        dataset.save_dataset()


def calculate_metrics(dataset, key=None):
    if dataset.dataset_name == 'BioGeneration':   # TODO: mask this line when testing on BioGeneration
        return 0.0, 0.0, 0.0
    correctness = []
    accuracy = []
    confidence = []
    for data in dataset.test_dataset:
        confidence += [np.exp(-l) for l in data[key + '_log_probs' if key is not None else 'log_probs']]
        if dataset.dataset_name.find('NLGraph') >= 0:
            cor = [-abs(int(data['correct_answer']) - e) if e is not None else -9999 for e in data[key + '_extracted_answers' if key is not None else 'extracted_answers']]
            correctness += cor
            accuracy += [c == 0 for c in cor]
        elif dataset.dataset_name == 'BioGeneration':
            cor = [fs if fs is not None else 0.0 for fs in data[key + '_factscore' if key is not None else 'factscore']]
            correctness += cor
            accuracy += [c >= 0.9 for c in cor]
        else:
            cor = [c[e] if e is not None else 0.0 for e, c in zip(data[key + '_extracted_answers' if key is not None else 'extracted_answers'], data['correctness'])]
            correctness += cor
            accuracy += [c == 1.0 for c in cor]

    # normalize
    if dataset.dataset_name.find('NLGraph') >= 0:
        mask = [c != -9999 for c in correctness]
        correctness = [c if c != -9999 else 0 for c in correctness]
        min_c = min(correctness)
        correctness = [c if m else min_c for m, c in zip(mask, correctness)]
    # remove correct responses
    correctness = [c for c, a in zip(correctness, accuracy) if not a]

    return np.mean(correctness), np.mean(accuracy), calculate_ece(np.array(accuracy), np.array(confidence))


def calculate_ece(y_true, y_pred_prob, n_bins=10):
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_pred_prob > bin_lower, y_pred_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def evaluate_grid_search(
        eval_strategy,
        preference_source,
        eval_source,
        dataset_name,
        preference_type,
        trainer_name='dpo',
        eval_model_name='gpt-4',
        top_p=0.5,
        filtered=True,
        load_from_exist=True,
        visualize=True
):
    preference_name = preference_source + '_' + preference_type
    if preference_type == 'direct':
        if filtered:
            preference_name += '_filtered_' + eval_model_name
        else:
            preference_name += '_' + eval_model_name
    elif preference_type == 'score':
        preference_name += '_' + str(top_p) + '_' + eval_model_name
    preference_name += '_' + trainer_name
    model_path = f'../output2/{dataset_name}/model/{preference_name}/'
    _grid_search_subdirs = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
    grid_search_subdirs = []
    for subdir in _grid_search_subdirs:
        if os.path.exists(os.path.join(model_path, subdir, 'log.json')) \
                and os.path.exists(os.path.join(model_path, subdir, 'adapter_model.safetensors')):
            grid_search_subdirs.append(subdir)
    if eval_strategy == 'latest':
        latest_dirs = {}
        for subdir in grid_search_subdirs:
            config, timestamp_str = subdir.rsplit('_timestamp=', 1)
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
            if config not in latest_dirs or timestamp > latest_dirs[config][0]:
                latest_dirs[config] = (timestamp, subdir)
        grid_search_subdirs = [info[1] for info in latest_dirs.values()]
    elif eval_strategy.find('best') >= 0:
        _, top_k = eval_strategy.split('_')
        top_k = int(top_k)
        val_losses = []
        for subdir in grid_search_subdirs:
            with open(os.path.join(model_path, subdir, 'log.json'), 'r', encoding='utf-8') as log_file:
                log_data = json.load(log_file)
                best_val_loss = log_data['best_val_loss']
                val_losses.append((best_val_loss, subdir))
        top_k_subdirs = nlargest(top_k, val_losses, key=lambda x: x[0])
        grid_search_subdirs = [subdir[1] for subdir in top_k_subdirs]

    response_path = f'../output2/{dataset_name}/response/{preference_name}/'
    metrics = {}
    for subdir in grid_search_subdirs:
        if not (load_from_exist and os.path.exists(os.path.join(response_path, subdir, f'{eval_source}.jsonl'))):
            dataset = load_dataset(
                dataset_name=dataset_name,
                model_name='',
                load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
            )
            dataset.load_test_path = os.path.join(response_path, subdir, f'{eval_source}.jsonl')
            if dataset_name == 'BioGeneration':
                dataset.response_sample_size = 3
            generate_evaluation_responses(dataset, f'../output2/{dataset_name}/model/{preference_name}/{subdir}')
        else:
            dataset = load_dataset(
                dataset_name=dataset_name,
                model_name='',
                load_test_path=os.path.join(response_path, subdir, f'{eval_source}.jsonl')
            )
        if visualize:
            with open(dataset.load_test_path[:-1], 'w', encoding='utf-8') as file:
                json.dump(dataset.test_dataset, file, indent=4)
        correctness, accuracy, ece = calculate_metrics(dataset)
        metrics[subdir + '_correctness'] = correctness
        metrics[subdir + '_accuracy'] = accuracy
        metrics[subdir + '_ece'] = ece
    os.makedirs(f'../output2/{dataset_name}/metric/{preference_name}/', exist_ok=True)
    with open(f'../output2/{dataset_name}/metric/{preference_name}/{eval_source}.jsonl', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=4)


def evaluate_original(dataset_name, eval_source, load_from_exist=True):
    if load_from_exist and os.path.exists(f'../output2/{dataset_name}/response/original/{eval_source}.jsonl'):
        return
    dataset = load_dataset(
        dataset_name,
        'llama-3',
        load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
    )
    dataset.load_test_path = f'../output2/{dataset_name}/response/original/{eval_source}.jsonl'
    generate_evaluation_responses(dataset, peft_dir=None)
    correctness, accuracy, ece = calculate_metrics(dataset)
    metrics = {
        'correctness': correctness,
        'accuracy': accuracy,
        'ece': ece
    }
    with open(f'../output2/{dataset_name}/metric/original/{eval_source}.jsonl', 'w', encoding='utf-8') as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--preference_source', type=str, default='all',
                        help='Source where preferences are collected: all, self')
    parser.add_argument('--eval_source', type=str, default='homogeneous',
                        help='Source where fine-tuned model will be evaluated: homogeneous')
    parser.add_argument('--dataset_name', type=str, default='KnowledgeCrosswords',
                        help='Name of the dataset: KnowledgeCrosswords, BioGeneration, CommonSense, NLGraph_SP')
    parser.add_argument('--eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model: gpt-4')
    parser.add_argument('--preference_type', type=str, default='oracle', help='Type of preference: oracle, direct, score')
    parser.add_argument('--trainer_name', type=str, default='dpo', help='Name of the trainer: dpo')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value: 0.5, 0.1')
    parser.add_argument('--filtered', type=bool, default=True,
                        help='Boolean flag to indicate if filtering is applied: True')
    parser.add_argument('--load_from_exist', type=bool, default=True)
    parser.add_argument('--eval_strategy', type=str, default='latest',
                        help='how many configs to use for evaluation: all, latest, best_n (n is int)')

    args = parser.parse_args()
    evaluate_grid_search(**vars(args))
