from preference_generation.metric import load_dataset
from preference_optimization.utils import *

import argparse
from netcal.metrics import ECE


def evaluate(
        dataset_name,
        eval_model_name,
        preference_type,
        trainer_name,
        top_p=0.5,
        filtered=True,
        load_from_exist=True
):
    preference_name = eval_model_name + '_' + preference_type
    if preference_type == 'direct':
        if filtered:
            preference_name += '_filtered'
    elif preference_type == 'score':
        preference_name += '_' + str(top_p)
    preference_name += '_' + trainer_name
    dataset = load_dataset(dataset_name, 'wow-llama-3', load_test=True)
    # TODO: only test in-domain data for now

    # generate responses
    if 'ori_responses' not in dataset.test_dataset[0]:
        dataset.generate_answer('CoT', 'test', 'ori')
        dataset.save_dataset()
    if not (load_from_exist and f'{preference_name}_responses' in dataset.test_dataset[0]):
        dataset.generate_answer('CoT', 'test', preference_name)
        dataset.save_dataset()

    # process responses
    eval_instruction_name = get_extract_instruction_name(dataset_name)
    if 'ori_extracted_answers' not in dataset.test_dataset[0]:
        dataset.process_answer('CoT', eval_instruction_name, 'test', 'ori')
        dataset.save_dataset()
    if not (load_from_exist and f'{preference_name}_extracted_answers' in dataset.test_dataset[0]):
        dataset.process_answer('CoT', eval_instruction_name, 'test', preference_name)
        dataset.save_dataset()

    calculate_accuracy(dataset, 'ori')
    calculate_accuracy(dataset, preference_name)


def calculate_accuracy(dataset, key):
    correctness = []
    accuracy = []
    confidence = []
    for data in dataset.test_dataset:
        confidence += data[key + '_log_probs']
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
    print(f'Correctness for {key} is: {sum(correctness) / len(correctness): .3f}')
    print(f'Accuracy for {key} is: {sum(accuracy) / len(accuracy): .3f}')
    ece = ECE(bins=10)
    print(f'ECE for {key} is: {ece.measure(confidence, accuracy): .3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script")

    parser.add_argument('dataset_name', type=str, default='KC', help='Name of the dataset')
    parser.add_argument('eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model')
    parser.add_argument('preference_type', type=str, default='direct', help='Type of preference (e.g., "direct")')
    parser.add_argument('trainer_name', type=str, default='dpo', help='Name of the trainer (e.g., "dpo")')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value (default: 0.5)')
    parser.add_argument('--filtered', type=bool, default=True,
                        help='Boolean flag to indicate if filtering is applied (default: True)')
    parser.add_argument('--load_from_exist', type=bool, default=True,
                        help='Boolean flag to indicate if loading from existing data is applied (default: True)')

    args = parser.parse_args()

    evaluate(
        args.dataset_name,
        args.eval_model_name,
        args.preference_type,
        args.trainer_name,
        top_p=args.top_p,
        filtered=args.filtered,
        load_from_exist=args.load_from_exist,
    )
