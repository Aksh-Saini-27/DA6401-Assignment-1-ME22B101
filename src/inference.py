"""
Inference Script
"""

import argparse
import numpy as np
import json
import os
import wandb

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.data_loader import load_data, one_hot_encode
from ann.neural_network import NeuralNetwork


def load_best_config():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    path = os.path.join(models_dir, "best_config.json")

    with open(path, "r") as f:
        return json.load(f)


def main():

    best_config = load_best_config()

    _, (X_test, y_test) = load_data(best_config["dataset"])

    input_size = X_test.shape[1]

    model = NeuralNetwork(cli_args=argparse.Namespace(**best_config),
                          input_size=input_size,
                          num_classes=10)

    model_path = os.path.join(os.path.dirname(__file__),
                              "..", "models", "best_model.npy")

    model.load_weights(model_path)

    logits = model.predict(X_test)

    loss = model.loss_fn.forward(logits, one_hot_encode(y_test))
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)


if __name__ == '__main__':
    main()