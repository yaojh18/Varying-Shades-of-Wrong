import json
import os
import re
import torch
import argparse
import networkx
import numpy as np
from datetime import datetime
from heapq import nlargest

from preference_generation.metric import load_dataset


def generate_evaluation_responses(dataset, peft_dir, load_from_exist=True):
    # generate responses
    response_flag = all(['responses' in data for data in dataset.test_dataset])
    extracted_answer_flag = all(['extracted_answers' in data for data in dataset.test_dataset])
    factscore_flag = all(['factscore' in data for data in dataset.test_dataset])
    if not load_from_exist or not response_flag:
        dataset.generate_answer(split='test', peft_dir=peft_dir)
        dataset.save_dataset()
        torch.cuda.empty_cache()
    # process responses
    if not load_from_exist or (not extracted_answer_flag and not factscore_flag):
        dataset.process_answer(split='test', peft_dir=peft_dir)
        dataset.save_dataset()
        torch.cuda.empty_cache()


def calculate_metrics(dataset, key=None):
    if dataset.dataset_name == 'NLGraph_SP':
        if 'normalizer' not in dataset.test_dataset[0]:
            if dataset.extract_instruction_name == 'shortest_path_extract':
                calculate_normalizer_for_shortest_path(dataset)
            elif dataset.extract_instruction_name == 'maximum_flow_extract':
                calculate_normalizer_for_maximum_flow(dataset)
    correctness = []
    accuracy = []
    confidence = []
    for data in dataset.test_dataset:
        confidence += [np.exp(-l) for l in data[key + '_log_probs' if key is not None else 'log_probs']]
        if dataset.dataset_name.find('NLGraph') >= 0:
            cor = [1.0 - (abs(e - data['correct_answer']) / (data['normalizer'] - data['correct_answer'])) if e is not None else 0.0
                   for e in data[key + '_extracted_answers' if key is not None else 'extracted_answers']]
            correctness += cor
            accuracy += [c == 1.0 for c in cor]
        elif dataset.dataset_name == 'BioGeneration':
            cor = [fs if fs is not None else 0.0 for fs in data[key + '_factscore' if key is not None else 'factscore']]
            correctness += cor
            accuracy += [c >= 0.9 for c in cor]
        else:
            cor = [c[e] if e is not None else 0.0 for e, c in zip(data[key + '_extracted_answers' if key is not None else 'extracted_answers'], data['correctness'])]
            correctness += cor
            accuracy += [c == 1.0 for c in cor]

    # remove correct responses

    wrong_correctness = [c for c, a in zip(correctness, accuracy) if not a]
    wrong_correctness = np.clip(wrong_correctness, a_min=0.0, a_max=1.0)
    correctness = np.clip(wrong_correctness, a_min=0.0, a_max=1.0)

    return np.mean(correctness), np.mean(wrong_correctness), np.mean(accuracy), calculate_ece(np.array(accuracy), np.array(confidence))


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


def find_longest_path(G, start, end):
    longest_length = 0

    def dfs(current_node, current_path, current_length):
        nonlocal longest_length

        if current_node == end:
            if current_length > longest_length:
                longest_length = current_length
            return

        for neighbor, data in G[current_node].items():
            if neighbor not in current_path:
                current_path.append(neighbor)
                dfs(neighbor, current_path, current_length + data['weight'])
                current_path.pop()

    dfs(start, [start], 0)
    return longest_length


def calculate_normalizer_for_shortest_path(dataset):
    graph_pattern = re.compile(r'an edge between node (\d+) and node (\d+) with weight (\d+)')
    question_pattern = re.compile(r'Q: Give the shortest path from node (\d+) to node (\d+)')
    for data in dataset.test_dataset:
        edge_strs = data['query'].split('\n')[1: -2]
        start, end = question_pattern.search(data['query'].split('\n')[-2]).groups()
        start, end = int(start), int(end)
        edge_args = [graph_pattern.search(edge_str).groups() for edge_str in edge_strs]
        edge_args = [(int(edge_arg[0]), int(edge_arg[1]), int(edge_arg[2])) for edge_arg in edge_args]
        G = networkx.Graph()
        G.add_weighted_edges_from(edge_args)
        data['normalizer'] = find_longest_path(G, start, end)


def calculate_normalizer_for_maximum_flow(dataset):
    for data in dataset.test_dataset:
        data['normalizer'] = 2 * data['correct_answer']


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
    if preference_type.find('direct') >= 0:
        if filtered:
            preference_name += '_filtered_' + eval_model_name
        else:
            preference_name += '_' + eval_model_name
    elif preference_type.find('score') >= 0:
        preference_name += '_' + str(top_p) + '_' + eval_model_name
    preference_name += '_' + trainer_name
    model_path = f'../output2/{dataset_name}/model/{preference_name}/'
    grid_search_subdirs = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
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
    for subdir in grid_search_subdirs:
        if not (load_from_exist and os.path.exists(os.path.join(response_path, subdir, f'{eval_source}.jsonl'))):
            if not os.path.exists(f'../output2/{dataset_name}/response/{eval_source}.jsonl'):
                print(f'No evaluation data for {dataset_name}/{eval_source}!')
                return
            dataset = load_dataset(
                dataset_name=dataset_name,
                model_name='',
                load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
            )
            dataset.load_test_path = os.path.join(response_path, subdir, f'{eval_source}.jsonl')
        else:
            dataset = load_dataset(
                dataset_name=dataset_name,
                model_name='',
                load_test_path=os.path.join(response_path, subdir, f'{eval_source}.jsonl')
            )
        # TODO: Warning: manually change some parameters
        if dataset_name == 'BioGeneration':
            dataset.response_sample_size = 3
        elif dataset_name == 'NLGraph_SP':
            if eval_source == 'homogeneous':
                dataset.extract_instruction_name = 'shortest_path_extract'
                dataset.extract_pattern = r'The total weight is (\d+)'
            elif eval_source == 'indomain':
                dataset.extract_instruction_name = 'maximum_flow_extract'
                dataset.extract_pattern = r'The maximum flow is (\d+)'

        generate_evaluation_responses(dataset, f'../output2/{dataset_name}/model/{preference_name}/{subdir}', load_from_exist)
        if visualize:
            with open(dataset.load_test_path[:-1], 'w', encoding='utf-8') as file:
                json.dump(dataset.test_dataset, file, indent=4)


def evaluate_original(dataset_name, eval_source, load_from_exist=True, visualize=True):
    if not(load_from_exist and os.path.exists(f'../output2/{dataset_name}/response/original/{eval_source}.jsonl')):
        dataset = load_dataset(
            dataset_name,
            'llama-3',
            load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
        )
        dataset.load_test_path = f'../output2/{dataset_name}/response/original/{eval_source}.jsonl'
    else:
        dataset = load_dataset(
            dataset_name,
            'llama-3',
            load_test_path=f'../output2/{dataset_name}/response/original/{eval_source}.jsonl'
        )
    if dataset_name == 'BioGeneration':
        dataset.response_sample_size = 3
    elif dataset_name == 'NLGraph_SP':
        if eval_source == 'homogeneous':
            dataset.extract_instruction_name = 'shortest_path_extract'
            dataset.extract_pattern = r'The total weight is (\d+)'
        elif eval_source == 'indomain':
            dataset.extract_instruction_name = 'maximum_flow_extract'
            dataset.extract_pattern = r'The maximum flow is (\d+)'
    if not (load_from_exist and 'responses' in dataset.test_dataset[0] and (
            'extracted_answers' in dataset.test_dataset[0] or 'factscore' in dataset.test_dataset[0])):
        generate_evaluation_responses(dataset, peft_dir=None)
    if visualize:
        with open(dataset.load_test_path[:-1], 'w', encoding='utf-8') as file:
            json.dump(dataset.test_dataset, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--preference_source', type=str, default='all',
                        help='Source where preferences are collected: all, self')
    parser.add_argument('--eval_source', type=str, default='homogeneous',
                        help='Source where fine-tuned model will be evaluated: homogeneous, indomain')
    parser.add_argument('--dataset_name', type=str, default='KnowledgeCrosswords',
                        help='Name of the dataset: KnowledgeCrosswords, BioGeneration, CommonSense, NLGraph_SP, MedMCQA, Science')
    parser.add_argument('--eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model: gpt-4')
    parser.add_argument('--preference_type', type=str, default='oracle',
                        help='Type of preference: oracle, direct, score, row, row_oracle, row_direct, row_score')
    parser.add_argument('--trainer_name', type=str, default='dpo', help='Name of the trainer: dpo')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value: 0.5, 0.1')
    parser.add_argument('--filtered', type=bool, default=True,
                        help='Boolean flag to indicate if filtering is applied: True')
    parser.add_argument('--load_from_exist', type=bool, default=True)
    parser.add_argument('--eval_strategy', type=str, default='latest',
                        help='how many configs to use for evaluation: all, latest, best_n (n is int)')

    args = parser.parse_args()
    evaluate_grid_search(**vars(args))
