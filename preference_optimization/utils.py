dataset_name_translator = {
    'KnowledgeCrosswords': 'KC',
    'BioGeneration': 'BioGeneration',
    'CommonSense': 'COM2',
    'NLGraph_SP': 'NLGraph_shortest_path'
}


def get_extract_instruction_name(dataset_name):
    if dataset_name == 'NLGraph_shortest_path':
        return 'shortest_path_extract'
    elif dataset_name == 'NLGraph_maximum_flow':
        return 'maximum_flow_extract'
    elif dataset_name == 'NLGraph_matching':
        return 'matching_extract'
    elif dataset_name == 'BioGeneration' or dataset_name == 'NLGraph':
        return 'nlgraph_extract'
    else:
        return 'multi_choice_extract'
