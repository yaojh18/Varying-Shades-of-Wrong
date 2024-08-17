export PYTHONPATH=$PYTHONPATH:$(pwd)
cd preference_optimization || exit
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source all --preference_type oracle
python3 evaluate.py --dataset_name BioGeneration --preference_source all --preference_type oracle
python3 evaluate.py --dataset_name CommonSense --preference_source all --preference_type oracle
python3 evaluate.py --dataset_name NLGraph_SP --preference_source all --preference_type oracle
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source all --preference_type direct
python3 evaluate.py --dataset_name BioGeneration --preference_source all --preference_type direct
python3 evaluate.py --dataset_name CommonSense --preference_source all --preference_type direct
python3 evaluate.py --dataset_name NLGraph_SP --preference_source all --preference_type direct
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source all --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name BioGeneration --preference_source all --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name CommonSense --preference_source all --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name NLGraph_SP --preference_source all --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source all --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name BioGeneration --preference_source all --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name CommonSense --preference_source all --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name NLGraph_SP --preference_source all --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source self --preference_type oracle
python3 evaluate.py --dataset_name BioGeneration --preference_source self --preference_type oracle
python3 evaluate.py --dataset_name CommonSense --preference_source self --preference_type oracle
python3 evaluate.py --dataset_name NLGraph_SP --preference_source self --preference_type oracle
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source self --preference_type direct
python3 evaluate.py --dataset_name BioGeneration --preference_source self --preference_type direct
python3 evaluate.py --dataset_name CommonSense --preference_source self --preference_type direct
python3 evaluate.py --dataset_name NLGraph_SP --preference_source self --preference_type direct
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source self --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name BioGeneration --preference_source self --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name CommonSense --preference_source self --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name NLGraph_SP --preference_source self --preference_type score --top_p 0.5
python3 evaluate.py --dataset_name KnowledgeCrosswords --preference_source self --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name BioGeneration --preference_source self --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name CommonSense --preference_source self --preference_type score --top_p 0.1
python3 evaluate.py --dataset_name NLGraph_SP --preference_source self --preference_type score --top_p 0.1

