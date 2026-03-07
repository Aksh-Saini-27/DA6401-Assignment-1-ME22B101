"""
Inference Script
"""

import argparse
import numpy as np
import json
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.data_loader import load_data, one_hot_encode
from ann.neural_network import NeuralNetwork


# =========================
# CLI
# =========================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset', default='mnist',
                        choices=['mnist','fashion_mnist'])

    parser.add_argument('-e','--epochs', type=int, default=10)
    parser.add_argument('-b','--batch_size', type=int, default=32)

    parser.add_argument('-l','--loss', default='cross_entropy',
                        choices=['mse','cross_entropy'])

    parser.add_argument('-o','--optimizer', default='sgd',
                        choices=['sgd','momentum','nag','rmsprop','adam','nadam'])

    parser.add_argument('-lr','--learning_rate', type=float, default=0.01)
    parser.add_argument('-wd','--weight_decay', type=float, default=0)

    parser.add_argument('-nhl','--num_layers', type=int, default=1)
    parser.add_argument('-sz','--hidden_size', type=int, nargs='+', default=[128])

    parser.add_argument('-a','--activation', default='relu',
                        choices=['relu','sigmoid','tanh'])

    parser.add_argument('-wi','--weight_init', default='random',
                        choices=['random','xavier'])

    parser.add_argument('-wp','--wandb_project', default='da6401_assignment_1')

    return parser.parse_args()


# =========================
# Load model
# =========================

def load_model(model_path):

    data = np.load(model_path, allow_pickle=True).item()

    return data


def main():

    args = parse_arguments()

    (_, _), (X_test, y_test) = load_data(args.dataset)

    model = NeuralNetwork(
        cli_args=args,
        input_size=X_test.shape[1],
        num_classes=10
    )

    src_dir = os.path.dirname(__file__)

    model_path = os.path.join(src_dir,"best_model.npy")

    weights = load_model(model_path)

    model.set_weights(weights)

    logits = model.predict(X_test)

    y_pred = np.argmax(logits,axis=1)

    acc = accuracy_score(y_test,y_pred)

    precision, recall, f1,_ = precision_recall_fscore_support(
        y_test,y_pred,average='macro'
    )

    conf_matrix = confusion_matrix(y_test,y_pred)

    print("Accuracy:",acc)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1-score:",f1)

    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()