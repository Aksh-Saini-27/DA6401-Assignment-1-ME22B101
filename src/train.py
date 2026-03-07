"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import os

# macOS / wandb safety
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# The following lines cause mutex freeze on macOS (gRPC + fork issue)
# Keeping them commented as requested (DO NOT DELETE)
# os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
# os.environ["WANDB_REQUIRE_SERVICE"] = "True"
# os.environ["WANDB_START_METHOD"] = "thread"

import argparse
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data, one_hot_encode, create_mini_batches
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


# =========================
# Argument Parser
# =========================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mse', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-wi', '--weight_init', type=str, default='random',
                        choices=['random', 'xavier', 'zeros'])

    return parser.parse_args()


# =========================
# Evaluation Function
# =========================

def evaluate(model, X, y_true):
    y_pred = []

    for i in range(0, X.shape[0], 128):
        batch_X = X[i:i+128]
        logits = model.predict(batch_X)
        y_pred.append(np.argmax(logits, axis=1))

    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)

    logits = model.predict(X)
    loss = model.loss_fn.forward(logits, one_hot_encode(y_true))

    return loss, acc


# =========================
# Main Training
# =========================

def main():
    args = parse_arguments()

    (X_train_full, y_train_full), (X_test, y_test) = load_data(args.dataset)

    import wandb

    run_name = (
        f"{args.dataset}_{args.optimizer}_"
        f"{args.activation}_h{args.num_layers}_"
        f"{'-'.join(map(str, args.hidden_size))}_"
        f"{args.weight_init}"
    )

    wandb.init(
        project="da6401_assignment_1",
        config=vars(args),
        name=run_name
    )

    # -------- Train/Validation split --------
    # Manual split to avoid joblib/sklearn multiprocessing hang on macOS
    np.random.seed(42)
    indices = np.random.permutation(len(X_train_full))
    val_size = int(len(X_train_full) * 0.1)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)

    # -------- Model --------
    model = NeuralNetwork(
        cli_args=args,
        input_size=X_train.shape[1],
        num_classes=10
    )

    opt_kwargs = {
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay
    }

    if args.optimizer == 'sgd':
        model.optimizer = SGD(**opt_kwargs)
    elif args.optimizer == 'momentum':
        model.optimizer = Momentum(**opt_kwargs)
    elif args.optimizer == 'nag':
        model.optimizer = NAG(**opt_kwargs)
    elif args.optimizer == 'rmsprop':
        model.optimizer = RMSProp(**opt_kwargs)
    elif args.optimizer == 'adam':
        model.optimizer = Adam(**opt_kwargs)
    elif args.optimizer == 'nadam':
        model.optimizer = Nadam(**opt_kwargs)

    best_val_acc = -1.0

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_model.npy")
    best_config_path = os.path.join(models_dir, "best_config.json")

    # -------- Training Loop --------

    for epoch in range(args.epochs):

        train_loss_epoch = 0
        train_steps = 0

        for batch_X, batch_y in create_mini_batches(
                X_train, y_train_oh, args.batch_size):

            y_pred = model.forward(batch_X)
            loss = model.loss_fn.forward(y_pred, batch_y)

            train_loss_epoch += loss
            train_steps += 1

            model.backward(batch_y, y_pred)
            model.update_weights()

        train_loss = train_loss_epoch / train_steps

        val_loss, val_acc = evaluate(model, X_val, y_val)
        _, train_acc = evaluate(model, X_train, y_train)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"Training complete! Best Validation Accuracy (this run): {best_val_acc:.4f}")

    # =========================
    # GLOBAL BEST CHECK
    # =========================

    global_best_acc = -1.0

    if os.path.exists(best_config_path):
        with open(best_config_path, "r") as f:
            old_config = json.load(f)
            global_best_acc = old_config.get("best_val_accuracy", -1.0)

    if best_val_acc > global_best_acc:
        print(f"New GLOBAL best model found! ({best_val_acc:.4f} > {global_best_acc:.4f})")

        model.save_weights(best_model_path)

        new_config = vars(args).copy()
        new_config["best_val_accuracy"] = best_val_acc

        with open(best_config_path, "w") as f:
            json.dump(new_config, f, indent=4)

    else:
        print(f"No global improvement. Current global best: {global_best_acc:.4f}")

    wandb.finish()


if __name__ == '__main__':
    # ==========================================
    # EXPERIMENT CODE FOR FASHION-MNIST
    # Comment out this block later and uncomment main() below
    # ==========================================
    # import sys
    # import os
    # os.environ["WANDB_MODE"] = "disabled" # Disable wandb to prevent fork issues on mac

    # configs = [
    #     {"name": "Config 1 (MNIST Best)", "args": ['-d', 'fashion_mnist', '-e', '10', '-o', 'rmsprop', '-a', 'relu', '-nhl', '1', '-sz', '128']},
    #     {"name": "Config 2 (Deeper + Adam)", "args": ['-d', 'fashion_mnist', '-e', '10', '-o', 'adam', '-a', 'relu', '-nhl', '2', '-sz', '128', '64']},
    #     {"name": "Config 3 (Deepest + Nadam)", "args": ['-d', 'fashion_mnist', '-e', '10', '-o', 'nadam', '-a', 'relu', '-nhl', '3', '-sz', '128', '64', '32']}
    # ]

    # for config in configs:
    #     print(f"\n--- Running {config['name']} ---")
    #     sys.argv = ['train.py'] + config['args']
    #     main()
        
    # Uncomment to run normally:
    main()















# """
# Main Training Script
# Entry point for training neural networks with command-line arguments
# """

# import os

# # macOS / wandb safety
# os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# # The following lines cause mutex freeze on macOS (gRPC + fork issue)
# # Keeping them commented as requested (DO NOT DELETE)
# os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
# os.environ["WANDB_REQUIRE_SERVICE"] = "True"
# os.environ["WANDB_START_METHOD"] = "thread"

# import argparse
# import numpy as np
# import json

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# from utils.data_loader import load_data, one_hot_encode, create_mini_batches
# from ann.neural_network import NeuralNetwork
# from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


# # =========================
# # Argument Parser
# # =========================

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Train a neural network')

#     parser.add_argument('-d', '--dataset', type=str, default='mnist',
#                         choices=['mnist', 'fashion_mnist'])
#     parser.add_argument('-e', '--epochs', type=int, default=10)
#     parser.add_argument('-b', '--batch_size', type=int, default=32)
#     parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
#                         choices=['mse', 'cross_entropy'])
#     parser.add_argument('-o', '--optimizer', type=str, default='sgd',
#                         choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
#     parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
#     parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
#     parser.add_argument('-nhl', '--num_layers', type=int, default=1)
#     parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
#     parser.add_argument('-a', '--activation', type=str, default='relu',
#                         choices=['relu', 'sigmoid', 'tanh'])
#     parser.add_argument('-wi', '--weight_init', type=str, default='random',
#                         choices=['random', 'xavier', 'zeros'])

#     return parser.parse_args()


# # =========================
# # Evaluation Function
# # =========================

# def evaluate(model, X, y_true):
#     y_pred = []

#     for i in range(0, X.shape[0], 128):
#         batch_X = X[i:i+128]
#         logits = model.predict(batch_X)
#         y_pred.append(np.argmax(logits, axis=1))

#     y_pred = np.concatenate(y_pred)
#     acc = accuracy_score(y_true, y_pred)

#     logits = model.predict(X)
#     loss = model.loss_fn.forward(logits, one_hot_encode(y_true))

#     return loss, acc


# # =========================
# # Main Training
# # =========================

# def main():
#     args = parse_arguments()

#     (X_train_full, y_train_full), (X_test, y_test) = load_data(args.dataset)

#     import wandb

#     run_name = (
#         f"{args.dataset}_{args.optimizer}_"
#         f"{args.activation}_h{args.num_layers}_"
#         f"{'-'.join(map(str, args.hidden_size))}_"
#         f"{args.weight_init}"
#     )

#     wandb.init(
#         project="da6401_assignment_1",
#         config=vars(args),
#         name=run_name
#     )

#     # -------- 2.1 Data Exploration Table --------
#     # table = wandb.Table(columns=[
#     #     "Image 1", "Image 2", "Image 3",
#     #     "Image 4", "Image 5", "Label"
#     # ])

#     # for label in range(10):
#     #     indices = np.where(y_train_full == label)[0]
#     #     imgs = [wandb.Image(X_train_full[idx].reshape(28, 28))
#     #             for idx in indices[:5]]

#     #     table.add_data(imgs[0], imgs[1], imgs[2],
#     #                    imgs[3], imgs[4], label)

#     # wandb.log({"Data Exploration Table": table})

#     # -------- Train/Validation split --------
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_full, y_train_full,
#         test_size=0.1,
#         random_state=42
#     )

#     y_train_oh = one_hot_encode(y_train)
#     y_val_oh = one_hot_encode(y_val)

#     # -------- Model --------
#     model = NeuralNetwork(
#         cli_args=args,
#         input_size=X_train.shape[1],
#         num_classes=10
#     )

#     opt_kwargs = {
#         'lr': args.learning_rate,
#         'weight_decay': args.weight_decay
#     }

#     if args.optimizer == 'sgd':
#         model.optimizer = SGD(**opt_kwargs)
#     elif args.optimizer == 'momentum':
#         model.optimizer = Momentum(**opt_kwargs)
#     elif args.optimizer == 'nag':
#         model.optimizer = NAG(**opt_kwargs)
#     elif args.optimizer == 'rmsprop':
#         model.optimizer = RMSProp(**opt_kwargs)
#     elif args.optimizer == 'adam':
#         model.optimizer = Adam(**opt_kwargs)
#     elif args.optimizer == 'nadam':
#         model.optimizer = Nadam(**opt_kwargs)

#     best_val_acc = -1.0

#     # -------- Training Loop --------

#     global_step = 0  # counts training iterations

#     for epoch in range(args.epochs):

#         train_loss_epoch = 0
#         train_steps = 0

#         for batch_X, batch_y in create_mini_batches(
#                 X_train, y_train_oh, args.batch_size):

#             y_pred = model.forward(batch_X)
#             loss = model.loss_fn.forward(y_pred, batch_y)

#             train_loss_epoch += loss
#             train_steps += 1

#             model.backward(batch_y, y_pred)

#             # =========================
#             # 2.9 Gradient Logging
#             # Log gradients of first 5 neurons
#             # Only for first 50 training iterations
#             # =========================
#             # if global_step < 50:
#             #     if hasattr(model.layers[0], 'grad_W'):
#             #         for n in range(5):
#             #             wandb.log({
#             #                 f"grad_neuron_{n}":
#             #                     np.mean(model.layers[0].grad_W[:, n])
#             #             }, step=global_step)

#             # global_step += 1

#             model.update_weights()

#         train_loss = train_loss_epoch / train_steps

#         val_loss, val_acc = evaluate(model, X_val, y_val)
#         _, train_acc = evaluate(model, X_train, y_train)

#         # Commented out activation logging for 2.9 (keeping as requested)
#         # for i, layer in enumerate(model.layers):
#         #     if hasattr(layer, 'output'):
#         #         wandb.log({
#         #             f"activation_layer_{i}":
#         #                 wandb.Histogram(layer.output)
#         #         })

#         wandb.log({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "train_accuracy": train_acc,
#             "val_loss": val_loss,
#             "val_accuracy": val_acc
#         })

#         print(
#             f"Epoch {epoch+1}/{args.epochs} - "
#             f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
#             f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
#         )

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             model.save_weights("model.npy")

#             with open("best_config.json", "w") as f:
#                 json.dump(vars(args), f, indent=4)

#             print(f"New best model saved (val_acc={val_acc:.4f})")

#     print(f"Training complete! Best Validation Accuracy: {best_val_acc:.4f}")

#     wandb.finish()


# if __name__ == '__main__':
#     main()