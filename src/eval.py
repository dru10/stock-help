import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import plots


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    f1 = 2 * precision * recall / (precision + recall)

    with open(os.path.join(model_path, "stats.txt"), "w") as f:
        f.write(f"N_Predictions = {len(preds)}\n\n")
        f.write(
            f"""
        |      |     Predicted       |
        |----------------------------|
        |      |   True  |   False   |
 -------|----------------------------|
        |True  |   {tp:3}   |  {fn:3}    |
 Actual |----------------------------|
        |False |   {fp:3}   |  {tn:3}    |
 -------|----------------------------|

"""
        )
        f.write(f"TP = {tp}\n")
        f.write(f"FN = {fn}\n")
        f.write(f"FP = {fp}\n")
        f.write(f"TN = {tn}\n\n")
        f.write(f"Accuracy  = {acc:.5f}\n")
        f.write(f"Precision = {precision:.5f}\n")
        f.write(f"Recall    = {recall:.5f}\n")
        f.write(f"F1        = {f1:.5f}\n")


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


def evaluate_predictions(
    symbol, model_path, pred, true, log_close, dates, thresh=0.5
):
    np.random.seed(42)

    ideal_moves = real_position(np.where(true > 0, 1, -1).reshape(-1))
    pred_moves = real_position(np.where(pred > thresh, 1, -1).reshape(-1))
    random_moves = real_position(
        np.random.choice([-1, 1], size=log_close.shape)
    )

    ideal_applied = np.multiply(log_close, ideal_moves)
    pred_applied = np.multiply(log_close, pred_moves)
    random_applied = np.multiply(log_close, random_moves)

    real_return = np.exp(np.cumsum(log_close))
    ideal_return = np.exp(np.cumsum(ideal_applied))
    strategy_return = np.exp(np.cumsum(pred_applied))
    random_return = np.exp(np.cumsum(random_applied))

    plots.returns(
        dates,
        real_return,
        [ideal_return],
        symbol,
        plot_destination=os.path.join(model_path, "ideal_returns.jpg"),
        labels=["Real", "Ideal"],
    )
    plots.returns(
        dates,
        real_return,
        [strategy_return, random_return],
        symbol,
        plot_destination=os.path.join(model_path, "strategy_return.jpg"),
        labels=["Real", "Strategy", "Random"],
    )
