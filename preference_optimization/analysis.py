import os
import json
import random
from preference_generation.metric import load_dataset
from preference_optimization.evaluate import calculate_metrics, calculate_normalizer_for_maximum_flow


def analysis(eval_source='homogeneous'):
    for dataset_name in ('KnowledgeCrosswords', 'BioGeneration', 'CommonSense', 'NLGraph_SP'):
        if not os.path.exists(f'../output2/{dataset_name}/response/{eval_source}.jsonl'):
            continue
        random.seed(42)
        raw_dataset = load_dataset(
            dataset_name=dataset_name,
            model_name='',
            load_test_path=f'../output2/{dataset_name}/response/{eval_source}.jsonl'
        )
        test_sample_idxs = [i for i in range(len(raw_dataset.test_dataset))]
        random.shuffle(test_sample_idxs)
        random.shuffle(test_sample_idxs)
        test_sample_idxs = test_sample_idxs[: len(raw_dataset.test_dataset) // 2]
        val_sample_idxs = [i for i in range(len(raw_dataset.test_dataset)) if i not in test_sample_idxs]
        parameter_names = [name for name in os.listdir(f'../output2/{dataset_name}/response/')
                           if os.path.isdir(f'../output2/{dataset_name}/response/{name}')]
        for parameter_name in parameter_names:
            if parameter_name != 'original':
                response_path = f'../output2/{dataset_name}/response/{parameter_name}/'
                grid_search_subdirs = [name for name in os.listdir(response_path) if
                                        os.path.isdir(os.path.join(response_path, name))]
            else:
                response_path = f'../output2/{dataset_name}/response/'
                grid_search_subdirs = ['original']
            metrics = {}
            best_correctness, best_wrong_correctness, best_accuracy, best_ece, best_grid_search_name = 0.0, 0.0, 0.0, 0.0, ''
            best_test_correctness, best_test_wrong_correctness, best_test_accuracy, best_test_ece = 0.0, 0.0, 0.0, 0.0
            for subdir in grid_search_subdirs:
                if not os.path.exists(os.path.join(response_path, subdir, f'{eval_source}.jsonl')):
                    continue
                dataset = load_dataset(
                    dataset_name=dataset_name,
                    model_name='',
                    load_test_path=os.path.join(response_path, subdir, f'{eval_source}.jsonl')
                )
                test_dataset = dataset.test_dataset
                if not ('extracted_answers' in dataset.test_dataset[0] or 'factscore' in dataset.test_dataset[0]):
                    continue
                dataset.test_dataset = [test_dataset[idx] for idx in val_sample_idxs]
                correctness, wrong_correctness, accuracy, ece = calculate_metrics(dataset)
                if correctness > best_correctness:
                    best_correctness = correctness
                    best_wrong_correctness = wrong_correctness
                    best_accuracy = accuracy
                    best_ece = ece
                    best_grid_search_name = subdir
                    dataset.test_dataset = [test_dataset[idx] for idx in test_sample_idxs]
                    correctness, wrong_correctness, accuracy, ece = calculate_metrics(dataset)
                    best_test_correctness = correctness
                    best_test_wrong_correctness = wrong_correctness
                    best_test_accuracy = accuracy
                    best_test_ece = ece

            metrics['best_correctness'] = best_correctness
            metrics['best_wrong_correctness'] = best_wrong_correctness
            metrics['best_accuracy'] = best_accuracy
            metrics['best_ece'] = best_ece
            metrics['best_grid_search_name'] = best_grid_search_name
            metrics['best_test_correctness'] = best_test_correctness
            metrics['best_test_wrong_correctness'] = best_test_wrong_correctness
            metrics['best_test_accuracy'] = best_test_accuracy
            metrics['best_test_ece'] = best_test_ece
            os.makedirs(f'../output2/{dataset_name}/metric/{parameter_name}/', exist_ok=True)
            with open(f'../output2/{dataset_name}/metric/{parameter_name}/{eval_source}.jsonl', 'w', encoding='utf-8') as file:
                json.dump(metrics, file, indent=4)


if __name__ == '__main__':
    analysis()
