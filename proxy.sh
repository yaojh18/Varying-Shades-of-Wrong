export PYTHONPATH=$PYTHONPATH:$(pwd)
cd preference_optimization || exit
datasets=("MedMCQA" "Science")
preference_sources=("all" "self")
top_p_values=("0.5" "0.1")

for dataset in "${datasets[@]}"; do
  for source in "${preference_sources[@]}"; do
      for top_p in "${top_p_values[@]}"; do
        python3 grid_search.py --dataset_name "$dataset" --preference_source "$source" --preference_type "score" --top_p "$top_p"
      done
  done
done

for dataset in "${datasets[@]}"; do
  for source in "${preference_sources[@]}"; do
      for top_p in "${top_p_values[@]}"; do
        python3 evaluate.py --dataset_name "$dataset" --preference_source "$source" --preference_type "score" --top_p "$top_p"
      done
  done
done