export PYTHONPATH=$PYTHONPATH:$(pwd)
cd preference_optimization || exit
datasets=("KnowledgeCrosswords" "BioGeneration" "CommonSense" "NLGraph_SP")
preference_sources=("all" "self")
preference_types=("oracle" "direct" "score")
top_p_values=("0.5" "0.1")
for dataset in "${datasets[@]}"; do
  for source in "${preference_sources[@]}"; do
    for type in "${preference_types[@]}"; do
      if [ "$type" == "score" ]; then
        for top_p in "${top_p_values[@]}"; do
          python3 finetune.py --dataset_name "$dataset" --preference_source "$source" --preference_type "$type" --top_p "$top_p"
        done
      else
        python3 finetune.py --dataset_name "$dataset" --preference_source "$source" --preference_type "$type"
      fi
    done
  done
done

