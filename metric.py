from dataset_NLGraph import NLGraph
from dataset_MMLU import MMLUPro
from dataset_KC import KnowledgeCrosswords
from dataset_FS import BioGeneration

import numpy as np
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
    gaps = []
    for prediction in predictions:
        # Calculate gaps for all possible pairs in B
        for i in range(len(prediction)):
            for j in range(i + 1, len(prediction)):
                gap = abs(prediction[i] - prediction[j])
                gaps.append(gap)
    threshold = np.quantile(gaps, 1 - top_p)

    if is_corrects is None:
        is_corrects = [[False] * len(predictions[0])] * len(predictions)

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

    accurate_count = 0
    all_count = 0
    for prediction, label, is_correct in zip(predictions, labels, is_corrects):
        for i in range(len(prediction)):
            for j in range(i + 1, len(prediction)):
                if not is_correct[i] and not is_correct[j] and abs(prediction[i] - prediction[j]) >= threshold and label[i] != label[j]:
                    all_count += 1
                    if (prediction[i] - prediction[j]) * (label[i] - label[j]) > 0:
                        accurate_count += 1
                    if detailed is not None:
                        i_bin_index = min(floor((label[i] - bin_start) / bin_size), detailed - 1)
                        j_bin_index = min(floor((label[j] - bin_start) / bin_size), detailed - 1)
                        detailed_all_count[i_bin_index][j_bin_index] += 1
                        if (prediction[i] - prediction[j]) * (label[i] - label[j]) > 0:
                            detailed_accurate_count[i_bin_index][j_bin_index] += 1

    print(f'Overall accuracy is {accurate_count / (all_count + 1e-8): .3f}')
    if detailed is not None:
        for i in range(detailed):
            for j in range(i + 1, detailed):
                print(
                    f'Accuracy for {j} over {i} is: {(detailed_accurate_count[i][j] + detailed_accurate_count[j][i]) / (detailed_all_count[i][j] + detailed_all_count[j][i] + 1e-8): .3f}')


def calculate_accuracy_pairwise():
    pass


def calculate_accuracy_log_prob(dataset_name, model_name, top_p=1.0):
    log_probs = []
    consistencies = []
    lengths = []
    labels = []
    is_corrects = []
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
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

    for data in dataset.train_dataset:
        log_probs.append([-l for e, l in zip(data['extracted answers'], data['log_probs']) if e is not None])
        consistency = [e for e in data['extracted answers'] if e is not None]
        counter = Counter(consistency)
        consistencies.append([counter[c] / len(consistency) for c in consistency])
        lengths.append([len(tokenizer(r)['input_ids']) for e, r in zip(data['extracted answers'], data['responses']) if e is not None])

    if dataset_name.find('NLGraph') >= 0:
        for data in dataset.train_dataset:
            labels.append([-abs(data['correct_answer'] - e) for e in data['extracted answers'] if e is not None])
            is_corrects.append([l == 0 for l in labels[-1]])
    elif dataset_name == 'BioGeneration':
        pass
    else:
        for data in dataset.train_dataset:
            labels.append([c[e] for e, c in zip(data['extracted answers'], data['correctness']) if e is not None])
            correct_answer = max(data['correctness'][0])
            is_corrects.append([l == correct_answer for l in labels[-1]])

    calculate_accuracy_score(log_probs, labels, is_corrects=is_corrects, top_p=top_p)
    calculate_accuracy_score(consistencies, labels, is_corrects=is_corrects, top_p=top_p)
    calculate_accuracy_score(lengths, labels, is_corrects=is_corrects, top_p=top_p)


if __name__ == '__main__':
    calculate_accuracy_log_prob('KC', 'llama-3', 1.0)
