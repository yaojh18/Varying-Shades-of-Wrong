conda activate base
python3 dataset_NLGraph.py --dataset_name NLGraph_shortest_path --extract_instruction_name shortest_path_extract
python3 dataset_NLGraph.py --dataset_name NLGraph_maximum_flow --extract_instruction_name maximum_flow_extract
python3 dataset_NLGraph.py --dataset_name NLGraph_matching --extract_instruction_name matching_extract