import math
import numpy as np


def is_nominal(meta, name):
    return meta[name][0] == "nominal"


def get_stats(data, meta):
    names = meta.names()
    stats = {'mean': {}, 'sd': {}}
    for name in names:
        if name == 'class' or is_nominal(meta, name):
            continue

        d = data[name]
        mean = np.sum(d) / float(len(d))
        vars = sum((val - mean) ** 2 for val in d)
        sd = math.sqrt(vars / float(len(d)))
        stats['mean'][name] = mean
        stats['sd'][name] = sd

    return stats


def normalize(stats, record, meta):
    names = meta.names()

    expected = 0 if record["class"] == meta["class"][-1][0] else 1
    arr = []
    for name in names:
        if name == "class":
            continue

        if is_nominal(meta, name):
            classes = meta[name][-1]
            val = record[name]
            k_hot = [0] * len(classes)
            k_hot[classes.index(val)] = 1
            arr.extend(k_hot)

        else:
            arr.append((record[name] - stats['mean'][name]) / stats['sd'][name])

    record = np.array(arr, ndmin=2)
    return record, expected


def calculate_f1(nn, stats, test_data, test_meta):
    """
        Returns F1 score and prints the predictions.
    """
    correct = tp = fp = fn = 0

    for record in test_data:
        record, expected = normalize(stats, record, test_meta)
        pred_score = nn.predict(record)
        pred = 1 if pred_score >= nn.SIGMOID_THRESHOLD else 0

        print "%.9f\t%d\t%d" % (pred_score, pred, expected)

        if pred == expected:
            correct += 1
            if expected == 1:
                tp += 1

        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, correct
