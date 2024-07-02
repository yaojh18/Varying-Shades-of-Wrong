import numpy as np


def calculate_accuracy_score(predictions, labels, label_num, top_p=1.0):
    assert isinstance(label_num, int)
    assert isinstance(top_p, float)
    gaps = []
    for prediction in predictions:
        # Calculate gaps for all possible pairs in B
        for i in range(len(prediction)):
            for j in range(i + 1, len(prediction)):
                gap = abs(prediction[i] - prediction[j])
                gaps.append(gap)
    threshold = np.quantile(gaps, top_p)

    count = [[0 for _ in range(label_num)] for i in range(label_num)]
    all_count = [[0 for _ in range(label_num)] for i in range(label_num)]
    for prediction, label in zip(predictions, labels):
        for i in range(label_num):
            for j in range(i + 1, label_num):
                i_idx = label.index(i)
                j_idx = label.index(j)
                if abs(prediction[j_idx] - prediction[i_idx]) > threshold:
                    all_count[i][j] += 1
                    if prediction[j_idx] - prediction[i_idx] > threshold:
                        count[i][j] += 1
    for i in range(label_num):
        for j in range(i + 1, label_num):
            print(f'Accuracy for {i} < {j} is {count[i][j] / (all_count[i][j] + 1e-8)}')

    print(sum([sum(c) for c in all_count]) / len(predictions) / 6)
