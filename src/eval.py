import csv
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_stats_to_txt(model_path, stats):
    with open(os.path.join(model_path, "stats.txt"), "w") as f:
        f.write(f"N_Predictions = {len(stats['preds'])}\n\n")
        f.write(
            f"""
        |      |     Predicted       |
        |----------------------------|
        |      |   True  |   False   |
 -------|----------------------------|
        |True  |   {stats['tp']:3}   |  {stats['fn']:3}    |
 Actual |----------------------------|
        |False |   {stats['fp']:3}   |  {stats['tn']:3}    |
 -------|----------------------------|

"""
        )
        f.write(f"TP = {stats['tp']}\n")
        f.write(f"FN = {stats['fn']}\n")
        f.write(f"FP = {stats['fp']}\n")
        f.write(f"TN = {stats['tn']}\n\n")
        f.write(f"Accuracy  = {stats['acc']:.5f}\n")
        f.write(f"Precision = {stats['precision']:.5f}\n")
        f.write(f"Recall    = {stats['recall']:.5f}\n")
        f.write(f"FPR       = {stats['fpr']:.5f}\n")
        f.write(f"TPR/FPR   = {stats['recall']/stats['fpr']:.5f}\n")
        f.write(f"F1        = {stats['f1']:.5f}\n")


def write_stats_to_csv(model_path, stats, filepath="results.csv"):
    _, model_type, symbol, _, lags, _, batch_size, _, epoch = os.path.normpath(
        model_path
    ).split(os.sep)

    values = OrderedDict(
        [
            ("ModelType", model_type),
            ("Symbol", symbol),
            ("Lags", lags),
            ("BatchSize", batch_size),
            ("Epoch", epoch),
            ("TP", stats["tp"]),
            ("TN", stats["tn"]),
            ("FP", stats["fp"]),
            ("TN", stats["tn"]),
            ("Accuracy", stats["acc"]),
            ("Precision", stats["precision"]),
            ("Recall", stats["recall"]),
            ("FPR", stats["fpr"]),
            ("TPR/FPR", stats["recall"] / stats["fpr"]),
            ("F1", stats["f1"]),
        ]
    )

    with open(filepath, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            # File is empty, write headers
            writer.writerow(values.keys())
        writer.writerow(values.values())


def calculate_stats(preds, y_test, model_path):
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, true in zip(preds, y_test):
        pred = pred.item()
        true = true.item()
        if pred == true:
            if true == 1:
                # True positive
                tp += 1
            else:
                # True negative
                tn += 1
        else:
            if true == 1:
                # False negative
                fn += 1
            else:
                # False positive
                fp += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)

    stats = {
        "preds": preds,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }

    write_stats_to_txt(model_path, stats)
    write_stats_to_csv(model_path, stats)


def predict_test_set(model, x_test, y_test):
    model.to(device)
    model.eval()
    test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False)

    preds = []
    with torch.no_grad():
        for input, label in test_loader:
            X = input.to(device)

            pred = model(X)

            preds.append(pred.item() >= 0.5)

    predictions = torch.tensor(preds, dtype=torch.float32).view(-1, 1)
    return predictions


def real_position(moves):
    total = 0
    new_pos = []
    for pos in moves:
        if total == 0 and pos == -1:
            new_pos.append(0)
        else:
            new_pos.append(pos)
            total += pos
    return np.array(new_pos)


def calculate_return(prediction, close, thresh=0.5):
    moves = real_position(np.where(prediction > thresh, 1, -1).reshape(-1))
    applied = np.multiply(close, moves)
    return np.exp(np.cumsum(applied))


def random_returns(close):
    np.random.seed(42)
    random_moves = real_position(np.random.choice([-1, 1], size=close.shape))
    random_applied = np.multiply(close, random_moves)
    return np.exp(np.cumsum(random_applied))


def evaluate_predictions(
    symbol, model_path, pred, true, log_close, dates, thresh=0.5
):
    ideal_return = calculate_return(true, log_close, 0)
    strategy_return = calculate_return(pred, log_close, thresh)
    random_return = random_returns(log_close)

    real_return = np.exp(np.cumsum(log_close))

    plots.returns(
        dates,
        real_return,
        [ideal_return],
        symbol,
        plot_destination=os.path.join(model_path, "ideal_returns.png"),
        labels=["Real", "Ideal"],
    )
    plots.returns(
        dates,
        real_return,
        [strategy_return, random_return],
        symbol,
        plot_destination=os.path.join(model_path, "strategy_return.png"),
        labels=["Real", "Strategy", "Random"],
    )


def evaluate_all_lags(
    dates, log_rets, lag_predictions, symbol, model_type, batch_size, epoch
):
    lag_rets = {
        lag: calculate_return(pred, log_rets)
        for lag, pred in lag_predictions.items()
    }
    real_return = np.exp(np.cumsum(log_rets))
    random_return = random_returns(log_rets)

    lag = next(iter(lag_predictions))
    try:
        combination = torch.zeros_like(lag_predictions[lag])
    except TypeError:
        combination = np.zeros_like(lag_predictions[lag])

    for idx in range(len(combination)):
        for key in lag_rets:
            combination[idx] += lag_rets[key][idx]
        combination[idx] /= len(lag_rets.keys())
    combination_return = calculate_return(combination, log_rets)

    plots.returns(
        dates,
        real_return,
        candidates=[lag_rets[key] for key in lag_rets]
        + [combination_return, random_return],
        symbol=symbol,
        plot_destination=os.path.join(
            "models",
            model_type,
            symbol,
            "lags",
            f"all_returns_batch_size_{batch_size}_epoch_{epoch}.png",
        ),
        labels=["Real"]
        + [f"{key} lags" for key in lag_rets]
        + ["Combination", "Random"],
    )
