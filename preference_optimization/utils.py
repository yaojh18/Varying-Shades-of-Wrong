dataset_name_translator = {
    'KnowledgeCrosswords': 'KC',
    'BioGeneration': 'BioGeneration',
    'CommonSense': 'COM2',
    'NLGraph_SP': 'NLGraph_shortest_path'
}


def get_extract_instruction_name_and_pattern(dataset_name):
    if dataset_name == 'NLGraph_shortest_path':
        return 'shortest_path_extract', r'The total weight is (\d+)'
    elif dataset_name == 'NLGraph_maximum_flow':
        return 'maximum_flow_extract', r'The maximum flow is (\d+)'
    elif dataset_name == 'NLGraph_matching':
        return 'matching_extract', r'The maximum number of matches is (\d+)'
    elif dataset_name == 'BioGeneration' or dataset_name == 'NLGraph':
        return 'nlgraph_extract', r'The final answer is (\d+)'
    else:
        return 'multi_choice_extract', r'([A-Z])(\.|\. .+)?$'
