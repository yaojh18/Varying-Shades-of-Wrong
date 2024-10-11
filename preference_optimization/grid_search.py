import argparse
import random
import time
from finetune import preference_optimization


def grid_search(
        learning_rate_range,
        lr_scheduler_type_range,
        weight_decay_range,
        num_train_epochs_range,
        beta_range,
        warmup_ratio_range,
        **kwargs
):
    for learning_rate in learning_rate_range:
        for lr_scheduler_type in lr_scheduler_type_range:
            for weight_decay in weight_decay_range:
                for num_train_epochs in num_train_epochs_range:
                    for beta in beta_range:
                        for warmup_ratio in warmup_ratio_range:
                            while True:
                                try:
                                    preference_optimization(
                                        learning_rate=learning_rate,
                                        lr_scheduler_type=lr_scheduler_type,
                                        weight_decay=weight_decay,
                                        num_train_epochs=num_train_epochs,
                                        beta=beta,
                                        warmup_ratio=warmup_ratio,
                                        **kwargs
                                    )
                                    break
                                except RuntimeError:
                                    print('unsloth breaks because of multi-gpu conflict.')
                                    r = random.Random(int(time.time()))
                                    time.sleep(r.randint(0, 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid Search Script")
    parser.add_argument('--preference_source', type=str, default='all',
                        help='Source where preferences are collected: all, self')
    parser.add_argument('--dataset_name', type=str, default='KnowledgeCrosswords',
                        help='Name of the dataset: KnowledgeCrosswords, BioGeneration, CommonSense, NLGraph_SP')
    parser.add_argument('--eval_model_name', type=str, default='gpt-4', help='Name of the evaluation model: gpt-4')
    parser.add_argument('--preference_type', type=str, default='oracle',
                        help='Type of preference: oracle, direct, score, row, row_oracle, row_direct, row_score.')
    parser.add_argument('--trainer_name', type=str, default='dpo',
                        help='Name of the trainer: dpo, rso, ipo, sppo, cpo, simpo, orpo')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p value: 0.5, 0.1')
    parser.add_argument('--filtered', type=bool, default=True,
                        help='Boolean flag to indicate if filtering is applied: True')
    parser.add_argument('--load_from_exist', type=bool, default=True)

    # training arguments
    parser.add_argument('--learning_rate_range', type=float, nargs='+', default=[1e-4, 5e-5, 1e-5],
                        help='Range of learning rates to explore: e.g., 1e-4 5e-5 1e-5')
    parser.add_argument('--lr_scheduler_type_range', type=str, nargs='+',
                        default=['cosine', 'cosine_with_restart', 'reduce_lr_on_plateau'],
                        help='Range of learning rate scheduler types: e.g. linear, cosine, cosine_with_restarts, reduce_lr_on_plateau')
    parser.add_argument('--weight_decay_range', type=float, nargs='+', default=[0, 1e-5, 1e-3],
                        help='Range of weight decay values: e.g., 0 1e-5 1e-3')
    parser.add_argument('--num_train_epochs_range', type=int, nargs='+', default=[1, 3, 5],
                        help='Range of number of training epochs: e.g., 1 3 5')
    parser.add_argument('--beta_range', type=float, nargs='+', default=[0.1],
                        help='Range of beta values: e.g., 0.1')
    parser.add_argument('--warmup_ratio_range', type=float, nargs='+', default=[0.1],
                        help='Range of warmup ratio: e.g., 0.1')
    args = parser.parse_args()

    grid_search(**vars(args))
