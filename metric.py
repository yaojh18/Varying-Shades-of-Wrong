from dataset_NLGraph import NLGraph
from dataset_MMLU import MMLUPro
from dataset_KC import KnowledgeCrosswords
from dataset_FS import BioGeneration
from utils import batch_query_openai, batch_query_open_sourced_llm

import numpy as np
import os
import json
import re
from math import floor
from collections import Counter
from transformers import AutoTokenizer
from huggingface_hub import login


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

    gaps = []
    corrects = []
    for prediction, label, is_correct in zip(predictions, labels, is_corrects):
        for i in range(len(prediction)):
            for j in range(i + 1, len(prediction)):
                if not is_correct[i] and not is_correct[j] and prediction[i] != prediction[j] and label[i] != label[j]:
                    gaps.append(abs(prediction[i] - prediction[j]))
                    if (prediction[i] - prediction[j]) * (label[i] - label[j]) > 0:
                        corrects.append(1)
                    else:
                        corrects.append(0)
                    if detailed is not None:
                        i_bin_index = min(floor((label[i] - bin_start) / bin_size), detailed - 1)
                        j_bin_index = min(floor((label[j] - bin_start) / bin_size), detailed - 1)
                        detailed_all_count[i_bin_index][j_bin_index] += 1
                        if (prediction[i] - prediction[j]) * (label[i] - label[j]) > 0:
                            detailed_accurate_count[i_bin_index][j_bin_index] += 1

    gaps = np.array(gaps)
    corrects = np.array(corrects)
    num_top_p = int(len(gaps) * top_p)
    top_p_indices = np.argsort(gaps)[-num_top_p:]
    print(f'Overall accuracy is {corrects[top_p_indices].mean(): .3f}')

    if detailed is not None:
        for i in range(detailed):
            for j in range(i + 1, detailed):
                print(f'Accuracy for {j} over {i} is: {(detailed_accurate_count[i][j] + detailed_accurate_count[j][i]) / (detailed_all_count[i][j] + detailed_all_count[j][i] + 1e-8): .3f}')


def load_dataset(dataset_name, model_name):
    if dataset_name.find('MC') >= 0 or dataset_name.find('KC') >= 0:
        dataset = KnowledgeCrosswords(
            dataset_name=dataset_name,
            model_name=model_name,
            knowledge=False,
            response_sample_size=10,
            dataset_sample_size=500,
            load_from_exist=True
        )
    elif dataset_name.find('NLGraph') >= 0:
        dataset = NLGraph(
            dataset_name=dataset_name,
            model_name=model_name,
            response_sample_size=10,
            dataset_sample_size=500,
            load_from_exist=True
        )
    elif dataset_name == 'BioGeneration':
        dataset = BioGeneration(
            dataset_name=dataset_name,
            model_name=model_name,
            response_sample_size=10,
            dataset_sample_size=500,
            load_from_exist=True
        )
    elif dataset_name == 'MMLUPro':
        dataset = MMLUPro(
            dataset_name=dataset_name,
            model_name=model_name,
            response_sample_size=10,
            dataset_sample_size=500,
            load_from_exist=True
        )
    else:
        raise NotImplementedError
    return dataset


def calculate_accuracy_from_response(dataset_name, model_name, top_p=1.0):
    log_probs = []
    consistencies = []
    lengths = []
    labels = []
    is_corrects = []
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    dataset = load_dataset(dataset_name, model_name)

    for data in dataset.train_dataset:
        log_probs.append([-l for e, l in zip(data['extracted answers'], data['log_probs']) if e is not None])
        consistency = [e for e in data['extracted answers'] if e is not None]
        counter = Counter(consistency)
        consistencies.append([counter[c] / len(consistency) for c in consistency])
        lengths.append([len(tokenizer(r)['input_ids']) for e, r in zip(data['extracted answers'], data['responses']) if
                        e is not None])

    if dataset_name.find('NLGraph') >= 0:
        for data in dataset.train_dataset:
            labels.append([-abs(int(data['correct_answer']) - e) for e in data['extracted answers'] if e is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        for data in dataset.train_dataset:
            labels.append([fs for fs in data['factscore'] if fs is not None])
            is_corrects.append([False for fs in data['factscore'] if fs is not None])
    else:
        for data in dataset.train_dataset:
            labels.append([c[e] for e, c in zip(data['extracted answers'], data['correctness']) if e is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=top_p)
    calculate_accuracy_score(consistencies, labels, is_corrects=is_corrects, top_p=top_p)
    calculate_accuracy_score(lengths, labels, is_corrects=is_corrects, top_p=top_p)


def calculate_accuracy_ask_llm_pairwise(
        dataset_name,
        model_name,
        evaluate_instruction_name,
        evaluate_model_name,
        load_from_exist=False
):
    if load_from_exist and os.path.exists(f'./output/pairwise/{model_name}/{dataset_name}.jsonl'):
        evaluation_jsonl = []
        with open(f'./output/pairwise/{model_name}/{dataset_name}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                evaluation_jsonl.append(json.loads(line.strip()))
    else:
        dataset = load_dataset(dataset_name, model_name)
        with open(f'instruction/{evaluate_instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        prompt_list = []
        evaluation_jsonl = []
        for idx, data in enumerate(dataset.train_dataset):
            responses = [r for r, e in zip(data['responses'], data['extracted answers']) if e is not None]
            if dataset_name.find('NLGraph') >= 0:
                label = [-abs(int(data['correct_answer']) - e) for e in data['extracted answers'] if e is not None]
                is_correct = [l == 0 for l in label]
            elif dataset_name == 'BioGeneration':
                label = [fs for fs in data['factscore'] if fs is not None]
                is_correct = [False for fs in data['factscore'] if fs is not None]
            else:
                label = [c[e] for e, c in zip(data['extracted answers'], data['correctness']) if e is not None]
                is_correct = [l == max(data['correctness'][0]) for l in label]
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    if not is_correct[i] and not is_correct[j] and label[i] != label[j]:
                        if dataset_name.find('MC') >= 0 or dataset_name == 'KC' or dataset_name == 'MMLUPro':
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
                            'content': instruction.format(data['query'], responses[i], responses[j])
                        }])
                        # TODO:
                        # prompt_list.append([{
                        #     'role': 'user',
                        #     'content': instruction.format(data['query'], responses[j], responses[i])
                        # }])
        if evaluate_model_name == 'gpt-4':
            lt_pairs, evaluations = batch_query_openai(prompt_list, model_name='gpt-4o', mode='evaluate')
        elif evaluate_model_name == 'gpt-3.5':
            lt_pairs, evaluations = batch_query_openai(prompt_list, model_name='gpt-3.5-turbo', mode='evaluate')
        elif evaluate_model_name == 'llama-3':
            lt_pairs, evaluations = batch_query_open_sourced_llm(prompt_list, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='evaluate')
        else:
            raise NotImplementedError
        # TODO:
        # for i in range(0, len(lt_pairs) // 2):
        #     preferred_output_1 = None
        #     preferred_output_2 = None
        #     for lt_pair in reversed(lt_pairs[i * 2]):
        #         if lt_pair[0] == '1' or lt_pair[0] == '2':
        #             preferred_output_1 = lt_pair
        #             break
        #     for lt_pair in reversed(lt_pairs[i * 2 + 1]):
        #         if lt_pair[0] == '1' or lt_pair[0] == '2':
        #             preferred_output_2 = lt_pair
        #             break
        #     if preferred_output_1 is None or preferred_output_2 is None:
        #         evaluation_jsonl[i]['evaluation'] = ''
        #         evaluation_jsonl[i]['extracted_evaluation'] = None
        #     elif preferred_output_1[0] != preferred_output_2[0] or preferred_output_1[1] > preferred_output_2[1]:
        #         evaluation_jsonl[i]c = evaluations[i * 2]
        #         evaluation_jsonl[i]['extracted_evaluation'] = int(preferred_output_1[0])
        #     else:
        #         evaluation_jsonl[i]['evaluation'] = evaluations[i * 2 + 1]
        #         evaluation_jsonl[i]['extracted_evaluation'] = 2 if int(preferred_output_1[0]) == 1 else 1
        pattern = re.compile(r'Preferred output: (\d+)')
        for evaluation, eval_json in zip(evaluations, evaluation_jsonl):
            eval_json['evaluation'] = evaluation
            match = re.search(pattern, evaluation)
            if match is not None:
                eval_json['extracted_evaluation'] = int(match.group(1))
            else:
                eval_json['extracted_evaluation'] = None

        os.makedirs(f'./output/pairwise/{model_name}/{evaluate_model_name}', exist_ok=True)
        with open(f'./output/pairwise/{model_name}/{evaluate_model_name}/{dataset_name}.jsonl', 'w',
                  encoding='utf-8') as file:
            for data in evaluation_jsonl:
                file.write(json.dumps(data) + '\n')

    accurate_count = 0
    all_count = 0
    for evaluation in evaluation_jsonl:
        if evaluation['extracted_evaluation'] is not None:
            all_count += 1
            if evaluation['extracted_evaluation'] == evaluation['ground_truth']:
                accurate_count += 1
    print(f'Overall accuracy is {accurate_count / (all_count + 1e-8): .3f}')


def calculate_accuracy_ask_llm_score(
        dataset_name,
        model_name,
        reward_name,
        evaluate_model_name,
        load_from_exist=False,
        top_p=1.0
):
    reward_name = evaluate_model_name + '_' + reward_name
    dataset = load_dataset(dataset_name, model_name)
    if not (load_from_exist and reward_name in dataset.train_dataset[0]):
        _, _, batch_size = reward_name.split('_')
        with open(f'instruction/evaluate_reward_{batch_size}.txt', encoding='utf-8') as f:
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
            _, evaluations = batch_query_open_sourced_llm(prompt_list, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='evaluate')
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
    for data in dataset.train_dataset:
        rewards.append([r for e, r in zip(data['extracted answers'], data[reward_name]) if e is not None and r is not None])

    if dataset_name.find('NLGraph') >= 0:
        for data in dataset.train_dataset:
            labels.append([-abs(int(data['correct_answer']) - e) for e, r in zip(data['extracted answers'], data[reward_name]) if e is not None and r is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        for data in dataset.train_dataset:
            labels.append([fs for fs in data['factscore'] if fs is not None])
            is_corrects.append([False for fs in data['factscore'] if fs is not None])
    else:
        for data in dataset.train_dataset:
            labels.append([c[e] for e, c, r in zip(data['extracted answers'], data['correctness'], data[reward_name]) if e is not None and r is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(rewards, labels, is_corrects=is_corrects, top_p=top_p)


if __name__ == '__main__':
    calculate_accuracy_ask_llm_pairwise('KC', 'gpt-3.5', 'evaluate_pairwise', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_score('KC', 'gpt-3.5', 'reward_5', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('MMLUPro', 'gpt-3.5', 'evaluate_pairwise', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_score('MMLUPro', 'gpt-3.5', 'reward_5', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_shortest_path', 'gpt-3.5', 'evaluate_pairwise', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_shortest_path', 'gpt-3.5', 'reward_5', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_maximum_flow', 'gpt-3.5', 'evaluate_pairwise', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_maximum_flow', 'gpt-3.5', 'reward_5', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_matching', 'gpt-3.5', 'evaluate_pairwise', 'gpt-4', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_matching', 'gpt-3.5', 'reward_5', 'gpt-4', load_from_exist=True)

    calculate_accuracy_ask_llm_pairwise('KC', 'gpt-3.5', 'evaluate_pairwise', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_score('KC', 'gpt-3.5', 'reward_5', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('MMLUPro', 'gpt-3.5', 'evaluate_pairwise', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_score('MMLUPro', 'gpt-3.5', 'reward_5', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_shortest_path', 'gpt-3.5', 'evaluate_pairwise', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_shortest_path', 'gpt-3.5', 'reward_5', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_maximum_flow', 'gpt-3.5', 'evaluate_pairwise', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_maximum_flow', 'gpt-3.5', 'reward_5', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_matching', 'gpt-3.5', 'evaluate_pairwise', 'gpt-3.5', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_matching', 'gpt-3.5', 'reward_5', 'gpt-3.5', load_from_exist=True)

    calculate_accuracy_ask_llm_pairwise('KC', 'gpt-3.5', 'evaluate_pairwise', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_score('KC', 'gpt-3.5', 'reward_5', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('MMLUPro', 'gpt-3.5', 'evaluate_pairwise', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_score('MMLUPro', 'gpt-3.5', 'reward_5', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_shortest_path', 'gpt-3.5', 'evaluate_pairwise', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_shortest_path', 'gpt-3.5', 'reward_5', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_maximum_flow', 'gpt-3.5', 'evaluate_pairwise', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_maximum_flow', 'gpt-3.5', 'reward_5', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_pairwise('NLGraph_matching', 'gpt-3.5', 'evaluate_pairwise', 'llama-3', load_from_exist=True)
    calculate_accuracy_ask_llm_score('NLGraph_matching', 'gpt-3.5', 'reward_5', 'llama-3', load_from_exist=True)
