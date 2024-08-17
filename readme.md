1. Create a new Conda environment using:
    ```bash
    conda create --name wow python=3.10 --file requirements.txt
    conda activate wow
    ```
2. Unzip required preference data under root dir.
3. Change training arguments under `./preference_opimization/finetune.py` from line 272. Please at least change the `per_device_train_batch_size` and `per_device_eval_batch_size` arguments to the appropriate size for your device to speed up training. It's also suggested to tune hyperparameters like lr, lr_scheduler here, since I didn't tune any hyperparameter.
4. Run `./finetune.sh` and then `./evaluae.sh`. Remember to give them executable permission by `chmod +x ./finetune.sh` and `chmod +x ./evaluae.sh`.