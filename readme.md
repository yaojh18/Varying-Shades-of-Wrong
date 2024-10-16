# Varing-Shades-of-Wrong Repository

This is the official repo for [Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only](https://arxiv.org/abs/2410.11055).

### Preparation
#### Environment
Setup the environment through:
```
conda env create -f wow.yaml
conda activate wow
```
#### Key
Before you run experiments, make sure you fill in the correct openai key and huggingface key in line 14 and 15 of `./preference_generation/utils.py`.

#### Data
Download the data we used for experiments through the [Google drive link](https://drive.google.com/file/d/1VPkqxG09vsWXlbvkN3wps5D_bnndXQY1/view?usp=drive_link). If you want to experiment with Bio Generation dataset, you need to further download data for FActScore dependency by:
```
python -m spacy download en_core_web_sm
python
>>> import nltk
>>> nltk.download('punkt_tab')
```
After downloading, unzip it under root dir. If you only want to use the experiment results, you can find the results we got for wrong-over-wrong preference eliciting and wrong-over-wrong alignment through this [Google drive link](https://drive.google.com/file/d/1S_OBn_6YYUQlPu1qDTxajQPweGiJio44/view?usp=drive_link).

### Wrong-over-wrong preference eliciting

This repo contains the implementation of constructing wrong-over-wrong data on 7 datasets and 3 LLMs. First, we need to sample responses for each (dataset, LLM) combination through:

```
python3 ./preference_generation/dataset.py
--dataset_name DATASET_STR
                                which dataset to use: "KC" (KnowledgeCrosswords), "NLGraph", "NLGraph_shortest_path", "NLGraph_maximum_flow", "NLGraph_matching", "BioGeneration", "MMLUPro", "COM2", "HellaSwag", "ChessPuzzle", "MedMCQA", "Science" (SciBench)
--model_name MODEL_STR
                                which model to use as generator: "llama-3", "gpt-3.5", "gpt-4"
--response_sample_size SIZE_INT
                                number of responses sampled for each question, default is 10
--response_sample_size SIZE_INT
                                number of sampled questions for each dataset, default is 625
--load_from_exist FLAG_BOOLEAN
                                whether to use existing processed dataset, dafault is True
--action ACTION_STR
                                what to do after query sampling: "g" for generating responses, "p" for parsering generated responses, "gp" for both, default is ""
```
After generation data, we will elicit wrong-over-wrong preference through 5 methods: heuristic, consistency-based, logits-based, pairwise comparison, score-based:
```
python3 ./preference_generation/metric.py
--dataset_name DATASET_STR
                               which dataset to elicit wrong-over-wrong preference: the same as above
--model_name MODEL_STR
                               generator name: the same as above
--strategy STRATEGY_STR
                               eliciting methods: 
                               "self" (output heuristic with margin 1.0, 0.5, 0.1, consistency-based with margin 1.0, 0.5, 0.1 and logits-based with margin 1.0, 0.5, 0.1 sequentially), 
                               "pair" (output pairwise comparison w/ and w/o consistency check sequentially), 
                               "score" (output score-based with margin 1.0, 0.5, 0.1 sequentially)
--eval_model_name SIZE_INT
                               evaluator name, only applying for "pair" and "score" strategy: the same as generator
--load_from_exist FLAG_BOOLEAN
                               whether to use elicited preferences, only applying for "pair" and "score" strategy, dafault is True
```
The output of wrong-over-wrong preference will be in `.\output\{MODEL_STR}\{DATASET_STR}.jsonl`, which contains the responses, parsered responses and elicited preferences.
### Wrong-over-wrong alignment
After obtaining wrong-over-wrong preferences, we can apply preference optimization methods for wrong-over-wrong alignment through:
```
python3 ./preference_optimization/finetune.py
--preference_source SOURCE_STR
                               source where preferences are collected: "self" (self generator) or "all" (mix generator)
--dataset_name DATASET_STR
                               name of the dataset: "KnowledgeCrosswords", "BioGeneration", "CommonSense", "NLGraph_SP", "MedMCQA", "Science"
--eval_model_name MODEL_STR
                               name of the evaluation model, default is "gpt-4": "llama-3", "gpt-3.5", "gpt-4"
--preference_type TYPE_STR
                               methods to elicit wrong-over-wrong preferences: "oracle", "direct", "score", "row" (wight-over-wrong alignment), "row_oracle" (right-over-wrong + wrong-over-wrong alignment oracle), "row_direct", "row_score".
--trainer_name TRAINER_STR 
                               preference opimization methods, default is "dpo": "dpo", "rso", "ipo", "sppo", "cpo", "simpo", "orpo"
--top_p P_INT
                               margin for score-based method, default is 0.5
--filtered F_BOOLEAN
                               whether apply consistenct check for pairwise comparison, default is True: True, False
--load_from_exist FLAG_BOOLEAN
                               whether to skip trained LLMs, dafault is True
```
If you want to apply grid search to find the best hyperparameters for wrong-over-wrong alignment, we have implemented it for you:
```
python3 ./preference_optimization/grid_search.py
--preference_source SOURCE_STR
--dataset_name DATASET_STR
--eval_model_name MODEL_STR
--preference_type TYPE_STR
--trainer_name TRAINER_STR 
--top_p P_INT
--filtered F_BOOLEAN
--load_from_exist FLAG_BOOLEAN

--learning_rate_range RANGE_ARGS
                                         learning rate list
--lr_scheduler_type_range SCHEDULER_ARGS
                                         learning rate scheduler list
--num_train_epochs_range EPOCH_ARGS
                                         number of training epochs list
--beta_range BETA_ARGS
                                         DPO's hyperparameter beta list
--warmup_ratio_range WARMUP_ARGS
                                         warmup ratio list
```
Before you evaluate the performance of wrong-over-wrong alignment, you should copy-and-paste the test dataset `.\output\{ANY_MODEL_STR}\{DATASET_STR}_test.jsonl` to `.\output2\{DATASET_STR}\response\` and rename it as `homogeneous.jsonl`. You can evaluate the performance of wrong-over-wrong alignment through :
```
python3 ./preference_optimization/evaluate.py
--preference_source SOURCE_STR
--dataset_name DATASET_STR
--eval_model_name MODEL_STR
--preference_type TYPE_STR
--trainer_name TRAINER_STR 
--top_p P_INT
--filtered F_BOOLEAN
--eval_source EVAL_SOURCE_STR
                                  name of the evaluation jsonl, default is "homogeneous".
--eval_strategy EVAL_STRATEGY_STR
                                  how many grid search results are use for evaluation, default is "latest": 
                                  "all" (evaluate all grid search results), 
                                  "latest" (evaluate all grid search results unless more than one grid searches only differ in timestamp and will evaluate the latest one), 
                                  "best_n" (n is int, only evaluate best n grid searches with the lowest val loss)
--load_from_exist FLAG_BOOLEAN
                                  whether to skip evaluted grid search results
```
The output of wrong-over-wrong alignment will be under `.\output2\{DATASET_STR}`. There will be 3 sub-dirs: `model` to store wrong-over-wrong aligned LLMs, `response` for the evaluation results of wrong-over-wrong alignment and `metric` for the calculated evaluation metrics.
### Citation
```
@misc{yao2024varyingshadeswrongaligning,
      title={Varying Shades of Wrong: Aligning LLMs with Wrong Answers Only}, 
      author={Jihan Yao and Wenxuan Ding and Shangbin Feng and Lucy Lu Wang and Yulia Tsvetkov},
      year={2024},
      eprint={2410.11055},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11055}, 
}
```

PRs are welcome for any issues or improvements. Any feedback on our work is welcomed!