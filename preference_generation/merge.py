import random
import re

from preference_generation.utils import *
from preference_generation.metric import *


def merge():
    file_name_list = ['KC_wo_knowledge', 'NLGraph_matching', 'NLGraph_maximum_flow', 'NLGraph_shortest_path']
    model_name_list = ['gpt-3.5', 'gpt-4', 'llama-3']

    for model_name in model_name_list:
        for file_name in file_name_list:
            train_dataset = []
            test_dataset = []
            with open(f'../output/{model_name}/{file_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    train_dataset.append(json.loads(line.strip()))
            with open(f'../output/{model_name}/{file_name}_test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    test_dataset.append(json.loads(line.strip()))

            merge_dataset = train_dataset + test_dataset
            merge_dataset = sorted(merge_dataset, key=lambda x: x['query'])

            with open(f'../output/{model_name}/{file_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in merge_dataset:
                    file.write(json.dumps(data) + '\n')
    file_name_list = ['KC', 'NLGraph_matching', 'NLGraph_maximum_flow', 'NLGraph_shortest_path']
    for model_name in model_name_list:
        for model_name2 in model_name_list:
            if model_name2 != 'llama-3':
                for file_name in file_name_list:
                    train_dataset = []
                    test_dataset = []
                    with open(f'../output/pairwise/{model_name}/{model_name2}/{file_name}.jsonl', 'r', encoding='utf-8') as file:
                        for line in file:
                            train_dataset.append(json.loads(line.strip()))
                    with open(f'../output/pairwise/{model_name}/{model_name2}/{file_name}_test.jsonl', 'r', encoding='utf-8') as file:
                        for line in file:
                            test_dataset.append(json.loads(line.strip()))

                    merge_dataset = train_dataset + test_dataset
                    merge_dataset = sorted(merge_dataset, key=lambda x: x['query'])

                    with open(f'../output/pairwise/{model_name}/{model_name2}/{file_name}.jsonl', 'w', encoding='utf-8') as file:
                        for data in merge_dataset:
                            file.write(json.dumps(data) + '\n')


def sample():
    file_name_list = ['NLGraph_matching', 'NLGraph_maximum_flow', 'NLGraph_shortest_path']
    model_name_list = ['gpt-3.5', 'gpt-4', 'llama-3']
    for model_name in model_name_list:
        merge_dataset = []
        for file_name in file_name_list:
            dataset = []
            with open(f'../output/{model_name}/{file_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    dataset.append(json.loads(line.strip()))
            merge_dataset += dataset[: 167]

        with open(f'../output/{model_name}/NLGraph.jsonl', 'w', encoding='utf-8') as file:
            for data in merge_dataset:
                file.write(json.dumps(data) + '\n')

    for model_name in model_name_list:
        val_query = []
        with open(f'../output/{model_name}/NLGraph.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                val_query.append(json.loads(line.strip())['query'])
        for model_name2 in model_name_list:
            if model_name2 != 'llama-3':
                dataset = []
                for file_name in file_name_list:
                    with open(f'../output/pairwise/{model_name}/{model_name2}/{file_name}.jsonl', 'r', encoding='utf-8') as file:
                        for line in file:
                            data = json.loads(line.strip())
                            if data['query'] in val_query:
                                dataset.append(json.loads(line.strip()))
                with open(f'../output/pairwise/{model_name}/{model_name2}/NLGraph.jsonl', 'w', encoding='utf-8') as file:
                    for data in dataset:
                        file.write(json.dumps(data) + '\n')


def sample_new_test_dataset():
    fs_train_queries = []
    fs_all_dataset = []
    fs_test_dataset = []
    with open(f'../output/gpt-3.5/BioGeneration.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            fs_train_queries.append(data['query'])
    with open(f'../dataset/KC.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            fs_all_dataset.append({
                'query': data['input'],
                'topic': data['topic']
            })
    for data in fs_all_dataset:
        if data['query'] not in fs_train_queries:
            fs_test_dataset.append(data)
    random.seed(42)
    fs_test_dataset = random.sample(fs_test_dataset, 125)
    with open(f'../output/gpt-3.5/BioGeneration_test.jsonl', 'w', encoding='utf-8') as file:
        for data in fs_test_dataset:
            file.write(json.dumps(data) + '\n')

    kc_train_queries = []
    kc_test_dataset = []
    with open(f'.../output/gpt-3.5/KC_wo_knowledge.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            kc_train_queries.append(data['query'])
    kc_dataset = KnowledgeCrosswords(
        dataset_name='KC',
        model_name='gpt-3.5',
        knowledge=False,
        response_sample_size=1,
        dataset_sample_size=-1,
        load_from_exist=False
    )
    for data in kc_dataset.dataset:
        if data['query'] not in kc_train_queries:
            kc_test_dataset.append(data)
    random.seed(42)
    kc_test_dataset = random.sample(kc_test_dataset, 125)
    with open(f'.../output/gpt-3.5/KC_wo_knowledge_test.jsonl', 'w', encoding='utf-8') as file:
        for data in kc_test_dataset:
            file.write(json.dumps(data) + '\n')

    nlgraph_train_queries = []
    with open(f'.../output/gpt-3.5/NLGraph.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            nlgraph_train_queries.append(data['query'])

    nlg_sp_test_dataset = []
    nlg_sp_dataset = NLGraph(
        dataset_name='NLGraph_shortest_path',
        model_name='gpt-3.5',
        response_sample_size=1,
        dataset_sample_size=-1,
        load_from_exist=False
    )
    for data in nlg_sp_dataset.dataset:
        if data['query'] not in nlgraph_train_queries:
            nlg_sp_test_dataset.append(data)
    random.seed(42)
    nlg_sp_test_dataset = random.sample(nlg_sp_test_dataset, 42)

    nlg_mf_test_dataset = []
    nlg_mf_dataset = NLGraph(
        dataset_name='NLGraph_maximum_flow',
        model_name='gpt-3.5',
        response_sample_size=1,
        dataset_sample_size=-1,
        load_from_exist=False
    )
    for data in nlg_mf_dataset.dataset:
        if data['query'] not in nlgraph_train_queries:
            nlg_mf_test_dataset.append(data)
    random.seed(42)
    nlg_mf_test_dataset = random.sample(nlg_mf_test_dataset, 42)

    nlg_mt_test_dataset = []
    nlg_mt_dataset = NLGraph(
        dataset_name='NLGraph_matching',
        model_name='gpt-3.5',
        response_sample_size=1,
        dataset_sample_size=-1,
        load_from_exist=False
    )
    for data in nlg_mt_dataset.dataset:
        if data['query'] not in nlgraph_train_queries:
            nlg_mt_test_dataset.append(data)
    random.seed(42)
    nlg_mt_test_dataset = random.sample(nlg_mt_test_dataset, 42)

    nlg_test_dataset = nlg_sp_test_dataset + nlg_mf_test_dataset + nlg_mt_test_dataset
    random.seed(42)
    random.shuffle(nlg_test_dataset)

    with open(f'.../output/gpt-3.5/NLGraph_test.jsonl', 'w', encoding='utf-8') as file:
        for data in nlg_test_dataset:
            file.write(json.dumps(data) + '\n')


def form_llama3_queries():
    dataset_name_list = ['KC', 'COM2', 'NLGraph', 'NLGraph_shortest_path', 'ChessPuzzle']
    model_name_list = ['gpt-3.5', 'gpt-4', 'llama-3']

    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            dataset = load_dataset(dataset_name, model_name)
            with open(f'.../instruction/evaluate_pairwise.txt', encoding='utf-8') as f:
                instruction = ''.join(f.readlines())
            prompt_list = []
            evaluation_jsonl = []
            idx = 0
            for data in dataset.train_dataset:
                responses = [r for r, e in zip(data['responses'], data['extracted answers']) if e is not None]
                if dataset_name.find('NLGraph') >= 0:
                    label = [-abs(int(data['correct_answer']) - e) for e in data['extracted answers'] if e is not None]
                    is_correct = [l == 0 for l in label]
                elif dataset_name == 'BioGeneration':
                    label = [fs for fs in data['factscore'] if fs is not None]
                    is_correct = [fs == 1.0 for fs in data['factscore'] if fs is not None]
                else:
                    label = [c[e] for e, c in zip(data['extracted answers'], data['correctness']) if e is not None]
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
                                'ground_truth': 1 if label[i] > label[j] else 2,
                                'evaluation': idx,
                                'reversed_evaluation': idx + 1
                            })
                            prompt_list.append({
                                'id': idx,
                                'prompt': instruction.format(data['query'], responses[i], responses[j]),
                                'output': None,
                            })
                            prompt_list.append({
                                'id': idx + 1,
                                'prompt': instruction.format(data['query'], responses[j], responses[i]),
                                'output': None,
                            })
                            idx += 2
            os.makedirs(f'../output/pairwise/{model_name}/llama-3', exist_ok=True)
            with open(f'../output/pairwise/{model_name}/llama-3/{dataset_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in evaluation_jsonl:
                    file.write(json.dumps(data) + '\n')
            os.makedirs(f'.../output/remote/short', exist_ok=True)
            with open(f'../output/remote/short/pairwise_{model_name}_llama-3_{dataset_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in prompt_list:
                    file.write(json.dumps(data) + '\n')

            with open(f'.../instruction/evaluate_reward_5.txt', encoding='utf-8') as f:
                instruction = ''.join(f.readlines())
            batch_size = 5
            idx = 0
            prompt_list = []
            for data in dataset.train_dataset:
                if 'choices' in data:
                    query = data['query'][:-9]
                else:
                    query = data['query']
                data['llama-3_reward_5_responses'] = []
                for i in range(0, len(data['responses']), batch_size):
                    args = [query]
                    for j in range(i, i + batch_size):
                        args.append(data['responses'][j])
                    data['llama-3_reward_5_responses'].append(idx)
                    prompt_list.append({
                        'id': idx,
                        'prompt': instruction.format(*args),
                        'output': None,
                    })
                    idx += 1
            dataset.save_dataset()
            os.makedirs(f'.../output/remote/long', exist_ok=True)
            with open(f'../output/remote/long/{model_name}_{dataset.output_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in prompt_list:
                    file.write(json.dumps(data) + '\n')


def process_llama3_queries():
    dataset_name_list = ['KC', 'COM2', 'NLGraph', 'NLGraph_shortest_path', 'ChessPuzzle']
    model_name_list = ['gpt-3.5', 'gpt-4', 'llama-3']
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            remote_responses = []
            evaluation_jsonl = []
            with open(f'../output/remote/short/pairwise_{model_name}_llama-3_{dataset_name}_output.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    remote_responses.append(json.loads(line.strip()))
            remote_responses.sort(key=lambda x: x['id'])
            with open(f'../output/pairwise/{model_name}/llama-3/{dataset_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    evaluation_jsonl.append(json.loads(line.strip()))
            pattern = re.compile(r'Preferred output: (\d+)')
            for eval_json in evaluation_jsonl:
                eval_json['evaluation'] = remote_responses[eval_json['evaluation']]['output']['content']
                match = re.search(pattern, eval_json['evaluation'])
                if match is not None:
                    eval_json['extracted_evaluation'] = int(match.group(1))
                else:
                    eval_json['extracted_evaluation'] = None
                eval_json['reversed_evaluation'] = remote_responses[eval_json['reversed_evaluation']]['output']['content']
                match = re.search(pattern, eval_json['reversed_evaluation'])
                if match is not None:
                    eval_json['reversed_extracted_evaluation'] = int(match.group(1))
                else:
                    eval_json['reversed_extracted_evaluation'] = None
            with open(f'../output/pairwise/{model_name}/llama-3/{dataset_name}.jsonl', 'w', encoding='utf-8') as file:
                for data in evaluation_jsonl:
                    file.write(json.dumps(data) + '\n')
            dataset = load_dataset(dataset_name, model_name)
            remote_responses = []
            with open(f'../output/remote/long/{model_name}_{dataset.output_name}_output.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    remote_responses.append(json.loads(line.strip()))
            remote_responses.sort(key=lambda x: x['id'])
            pattern = re.compile(r"Score: (\d+)")
            for data in dataset.train_dataset:
                rewards = []
                responses = []
                for res_idx in data['llama-3_reward_5_responses']:
                    responses.append(remote_responses[res_idx]['output']['content'])
                    scores = re.findall(pattern, responses[-1])
                    scores = list(map(int, scores))
                    if len(scores) != 5:
                        scores = [None] * 5
                    rewards += scores
                data['llama-3_reward_5_responses'] = responses
                data['llama-3_reward_5'] = rewards
            dataset.save_dataset()


if __name__ == '__main__':
    pass
