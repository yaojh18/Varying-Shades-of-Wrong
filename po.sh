export PYTHONPATH=$PYTHONPATH:$(pwd)
cd preference_optimization || exit
pos=("ipo" "sppo" "simpo" "orpo")
preference_sources=("all" "self")
preference_types=("oracle" "direct" "score")
top_p_values=("0.5" "0.1")
for po in "${pos[@]}"; do
  for source in "${preference_sources[@]}"; do
    for type in "${preference_types[@]}"; do
      if [ "$type" == "score" ]; then
        for top_p in "${top_p_values[@]}"; do
          python3 finetune.py --dataset_name "NLGraph_SP" --trainer_name "$po" --preference_source "$source" --preference_type "$type" --top_p "$top_p"
        done
      else
        python3 finetune.py --dataset_name "NLGraph_SP" --trainer_name "$po" --preference_source "$source" --preference_type "$type"
      fi
    done
  done
done

for po in "${pos[@]}"; do
  for source in "${preference_sources[@]}"; do
    for type in "${preference_types[@]}"; do
      if [ "$type" == "score" ]; then
        for top_p in "${top_p_values[@]}"; do
          python3 evaluate.py --dataset_name "NLGraph_SP" --trainer_name "$po" --preference_source "$source" --preference_type "$type" --top_p "$top_p"
        done
      else
        python3 evaluate.py --dataset_name "NLGraph_SP" --trainer_name "$po" --preference_source "$source" --preference_type "$type"
      fi
    done
  done
done

