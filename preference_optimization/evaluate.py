import json
import os

from preference_generation.metric import load_dataset
from preference_optimization.utils import *

import argparse
import numpy as np


def evaluate(
        preference_source,
        eval_source,
        dataset_name,
        preference_type,
        trainer_name='dpo',
        eval_model_name='gpt-4',
        top_p=0.5,
        filtered=True,
        load_from_exist=True
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
    dataset = load_dataset(
        dataset_name_translator[dataset_name],
        'llama-3',
        load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
    )

    # generate responses
    if 'ori_responses' not in dataset.test_dataset[0]:
        dataset.generate_answer('CoT', 'test', 'ori')
        dataset.save_dataset()
    if not (load_from_exist and f'{preference_name}_responses' in dataset.test_dataset[0]):
        dataset.generate_answer('CoT', 'test', preference_name)
        dataset.save_dataset()
    if dataset_name != 'BioGeneration':
        # process responses
        eval_instruction_name = get_extract_instruction_name(dataset_name)
        if 'ori_extracted_answers' not in dataset.test_dataset[0]:
            dataset.process_answer('CoT', eval_instruction_name, 'test', 'ori')
            dataset.save_dataset()
        if not (load_from_exist and f'{preference_name}_extracted_answers' in dataset.test_dataset[0]):
            dataset.process_answer('CoT', eval_instruction_name, 'test', preference_name)
            dataset.save_dataset()

        ori_correctness, ori_accuracy, ori_ece = calculate_metrics(dataset, 'ori')
        new_correctness, new_accuracy, new_ece = calculate_metrics(dataset, preference_name)
        if not os.path.exists(f'../output2/{dataset_name}/metric/{eval_source}.jsonl'):
            with open(f'../output2/{dataset_name}/metric/{eval_source}.jsonl', 'w', encoding='utf-8') as file:
                json.dump({}, file)
        with open(f'../output2/{dataset_name}/metric/{eval_source}.jsonl', 'r', encoding='utf-8') as file:
            metrics = json.load(file)
            metrics['ori_correctness'] = ori_correctness
            metrics['ori_accuracy'] = ori_accuracy
            metrics['ori_ece'] = ori_ece
            metrics[f'{preference_name}_correctness'] = new_correctness
            metrics[f'{preference_name}_accuracy'] = new_accuracy
            metrics[f'{preference_name}_ece'] = new_ece
        with open(f'../output2/{dataset_name}/metric/{eval_source}.jsonl', 'w', encoding='utf-8') as file:
            json.dump(metrics, file)


def calculate_metrics(dataset, key):
    correctness = []
    accuracy = []
    confidence = []
    for data in dataset.test_dataset:
        confidence += [np.exp(-l) for l in data[key + '_log_probs']]
        if dataset.dataset_name.find('NLGraph') >= 0:
            cor = [-abs(int(data['correct_answer']) - e) if e is not None else -9999 for e in data[key + '_extracted_answers']]
            correctness += cor
            accuracy += [c == 0 for c in cor]
        elif dataset.dataset_name == 'BioGeneration':
            cor = [fs if fs is not None else 0.0 for fs in data[key + '_factscore']]
            correctness += cor
            accuracy += [c >= 0.9 for c in cor]
        else:
            cor = [c[e] if e is not None else 0.0 for e, c in zip(data[key + '_extracted_answers'], data['correctness'])]
            correctness += cor
            accuracy += [c == 1.0 for c in cor]

    # normalize
    if dataset.dataset_name.find('NLGraph') >= 0:
        mask = [c != -9999 for c in correctness]
        correctness = [c if c != -9999 else 0 for c in correctness]
        max_c = max(correctness)
        min_c = min(correctness)
        correctness = [(c - min_c) / (max_c - min_c) if m else 0.0 for m, c in zip(mask, correctness)]
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

    args = parser.parse_args()

    evaluate(
        preference_source=args.preference_source,
        eval_source=args.eval_source,
        dataset_name=args.dataset_name,
        preference_type=args.preference_type,
        trainer_name=args.trainer_name,
        eval_model_name=args.eval_model_name,
        top_p=args.top_p,
        filtered=args.filtered,
        load_from_exist=args.load_from_exist,
    )
