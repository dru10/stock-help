import torch
import os
from torch.utils.data import DataLoader, TensorDataset

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
