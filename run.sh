python3 dataset_KC.py --dataset_name MC_easy --model_name gpt-3.5
python3 dataset_KC.py --dataset_name MC_medi --model_name gpt-3.5
python3 dataset_KC.py --dataset_name MC_hard --model_name gpt-3.5
python3 dataset_FS.py --model_name gpt-3.5
python3 dataset_MMLU.py --model_name gpt-3.5
python3 dataset_NLGraph.py --dataset_name NLGraph_shortest_path --extract_instruction_name shortest_path_extract --model_name gpt-3.5
python3 dataset_NLGraph.py --dataset_name NLGraph_maximum_flow --extract_instruction_name maximum_flow_extract --model_name gpt-3.5
python3 dataset_NLGraph.py --dataset_name NLGraph_matching --extract_instruction_name matching_extract --model_name gpt-3.5