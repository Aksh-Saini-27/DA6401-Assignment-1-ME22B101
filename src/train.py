"""
Main Training Script
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import argparse
import numpy as np
import json
import wandb

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data, one_hot_encode, create_mini_batches
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


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
# Evaluation
# =========================

def evaluate(model, X, y):

    logits = model.predict(X)

    y_pred = np.argmax(logits,axis=1)

    acc = accuracy_score(y,y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='macro'
    )

    return acc, precision, recall, f1


# =========================
# Optimizer helper
# =========================

def get_optimizer(name, lr, wd):

    if name == 'sgd':
        return SGD(lr=lr, weight_decay=wd)

    if name == 'momentum':
        return Momentum(lr=lr, weight_decay=wd)

    if name == 'nag':
        return NAG(lr=lr, weight_decay=wd)

    if name == 'rmsprop':
        return RMSProp(lr=lr, weight_decay=wd)

    if name == 'adam':
        return Adam(lr=lr, weight_decay=wd)

    if name == 'nadam':
        return Nadam(lr=lr, weight_decay=wd)


# =========================
# Training
# =========================

def main():

    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    (X_train_full, y_train_full), (X_test, y_test) = load_data(args.dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        random_state=42
    )

    y_train_oh = one_hot_encode(y_train)

    model = NeuralNetwork(
        cli_args=args,
        input_size=X_train.shape[1],
        num_classes=10
    )

    model.optimizer = get_optimizer(
        args.optimizer,
        args.learning_rate,
        args.weight_decay
    )

    best_f1 = -1

    src_dir = os.path.dirname(__file__)

    model_path = os.path.join(src_dir,"best_model.npy")
    config_path = os.path.join(src_dir,"best_config.json")

    for epoch in range(args.epochs):

        for batch_X, batch_y in create_mini_batches(
                X_train, y_train_oh, args.batch_size):

            logits = model.forward(batch_X)

            loss = model.loss_fn.forward(logits, batch_y)

            model.backward(batch_y, logits)

            model.update_weights()

        train_acc,_,_,_ = evaluate(model,X_train,y_train)
        val_acc,_,_,val_f1 = evaluate(model,X_val,y_val)

        wandb.log({
            "epoch":epoch,
            "train_acc":train_acc,
            "val_acc":val_acc,
            "val_f1":val_f1
        })

        print(f"Epoch {epoch+1}/{args.epochs} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_weights = model.get_weights()

            np.save(model_path, best_weights)

            with open(config_path,"w") as f:
                json.dump(vars(args),f,indent=4)

    print("Best F1:",best_f1)

    wandb.finish()


if __name__ == "__main__":
    main()