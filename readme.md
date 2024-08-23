## 8 / 23
1. Unzip required preference data `./output2` under root dir. Remember to clean the original `./output2` folder since `load_from_exsist` is set to be true for `./preference_optimization/finetune.py` and `./preference_optimization/evaluate.py`.
2. The IO system I implemented doesn't support jsonl file with indent. Thus, I save a json file with indent under the same path, which is used to visualize data of the corresponding jsonl file with the same name.
3. The organization of the output files are: `./output2/{dataset_name}/{model, response or metric}/{main experiment setup}/{grid search parameters}/{necessary files, like *.safetensor, log.json, homogeneous.jsonl, homogeneous.json, etc.}`. For "metric", all the "grid search parameters" are gathered in a single file.
4. How to run finetune and evaluate experiment? The design of grid search and evaluation of grid search is serial. Still, you can parallelize the experiment on "main experiment setup", specifically, `dataset`, `preference_source`, `preference_type`, `top_p`. Please refer to `./finetune.sh` and `./evaluate.sh` for example. DO NOT run experiment on NLGraph yet!!! I haven't fix the less wrong metric issue.
5. Please go to `./preference_opimization/finetune.py` from line 391 to line 401 to change the grid search hyperparameters. If you think grid searching on a certain hyperparameter is unnecessary, just make it a list with a single element.
6. There are 3 `eval_strategy` for `./preference_opimization/evaluate.py`. 'all' for evaluating all grid search results under the same experiment setup. 'latest', which is default, will evaluate all grid search results under the same experiment unless more than one grid searches only differ in timestamp and will evaluate the latest one. This is because even when setting `load_from_exist` to be false, new grid search will not overwrite the original file because timestamp is different. `best_n`, where n is a int number, is a time-efficient method that will only evaluate best n grid searches will the lowest val loss.

## 8 / 16
1. Create a new Conda environment using:
    ```bash
    conda create --name wow python=3.10
    conda activate wow
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
   Please install pytorch with the correct cuda version (11.8 or 12.1)
2. Unzip required preference data under root dir.
3. Change training arguments under `./preference_opimization/finetune.py` from line 272. Please at least change the `per_device_train_batch_size` and `per_device_eval_batch_size` arguments to the appropriate size for your device to speed up training. It's also suggested to tune hyperparameters like lr, lr_scheduler here, since I didn't tune any hyperparameter.
4. Run `./finetune.sh` and then `./evaluate.sh`. Remember to give them executable permission by `chmod +x ./finetune.sh` and `chmod +x ./evaluate.sh`.