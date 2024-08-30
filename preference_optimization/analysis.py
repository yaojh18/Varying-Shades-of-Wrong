import os
import json
from preference_generation.metric import load_dataset
from preference_optimization.evaluate import calculate_metrics


def analysis():
    for dataset_name in ('KnowledgeCrosswords', 'BioGeneration', 'CommonSense', 'NLGraph_SP'):
        for parameter_name in ('all_direct_filtered_gpt-4_dpo', 'all_oracle_dpo', 'all_score_0.1_gpt-4_dpo', 'all_score_0.5_gpt-4_dpo', 'self_direct_filtered_gpt-4_dpo', 'self_oracle_dpo', 'self_score_0.1_gpt-4_dpo', 'self_score_0.5_gpt-4_dpo'):
            response_path = f'../output2/{dataset_name}/response/{parameter_name}/'
            grid_search_subdirs = [name for name in os.listdir(response_path) if
                                    os.path.isdir(os.path.join(response_path, name))]
            metrics = {}
            best_correctness, best_wrong_correctness, best_accuracy, best_ece, best_grid_search_name = 0.0, 0.0, 0.0, 0.0, ''
            best_test_wrong_correctness, best_test_accuracy, best_test_ece = 0.0, 0.0, 0.0
            for subdir in grid_search_subdirs:
                dataset = load_dataset(
                    dataset_name=dataset_name,
                    model_name='',
                    load_test_path=os.path.join(response_path, subdir, f'homogeneous.jsonl')
                )
                test_dataset = dataset.test_dataset
                if not ('extracted_answers' in dataset.test_dataset[0] or 'factscore' in dataset.test_dataset[0]):
                    continue
                dataset.test_dataset = test_dataset[: len(test_dataset) // 2]
                correctness, wrong_correctness, accuracy, ece = calculate_metrics(dataset)
                if correctness > best_correctness:
                    best_correctness = correctness
                    best_wrong_correctness = wrong_correctness
                    best_accuracy = accuracy
                    best_ece = ece
                    best_grid_search_name = subdir
                    dataset.test_dataset = test_dataset[len(test_dataset) // 2:]
                    _, wrong_correctness, accuracy, ece = calculate_metrics(dataset)
                    best_test_wrong_correctness = wrong_correctness
                    best_test_accuracy = accuracy
                    best_test_ece = ece

            metrics['best_correctness'] = best_correctness
            metrics['best_wrong_correctness'] = best_wrong_correctness
            metrics['best_accuracy'] = best_accuracy
            metrics['best_ece'] = best_ece
            metrics['best_grid_search_name'] = best_grid_search_name
            metrics['best_test_wrong_correctness'] = best_test_wrong_correctness
            metrics['best_test_accuracy'] = best_test_accuracy
            metrics['best_test_ece'] = best_test_ece
            os.makedirs(f'../output2/{dataset_name}/metric/{parameter_name}/', exist_ok=True)
            with open(f'../output2/{dataset_name}/metric/{parameter_name}/homogeneous.jsonl', 'w', encoding='utf-8') as file:
                json.dump(metrics, file, indent=4)


if __name__ == '__main__':
    analysis()
