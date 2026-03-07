# Multi-Layer Perceptron (MLP) -- NumPy Implementation

This project implements a fully connected neural network (Multi-Layer
Perceptron) entirely from scratch using **NumPy**.\
No deep learning frameworks such as TensorFlow or PyTorch were used.

The objective of this project is to build a strong understanding of
neural network internals, including forward propagation,
backpropagation, gradient computation, optimization algorithms, and
performance evaluation.

The implementation supports both **MNIST** and **Fashion-MNIST**
datasets and allows flexible selection of activation functions, loss
functions, optimizers, and weight initialization strategies.

------------------------------------------------------------------------

## Project Structure

    ├── models/
    │   ├── best_config.json
    │   └── best_model.npy
    │
    ├── src/
    │   ├── ann/
    │   │   ├── activations.py
    │   │   ├── neural_layer.py
    │   │   ├── neural_network.py
    │   │   ├── objective_functions.py
    │   │   └── optimizers.py
    │   │
    │   ├── utils/
    │   │   └── data_loader.py
    │   │
    │   ├── train.py
    │   └── inference.py
    │
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

##  Features

### Activation Functions

-   Sigmoid\
-   Tanh\
-   ReLU

### Loss Functions

-   Mean Squared Error (MSE)\
-   Cross-Entropy (with numerically stable softmax)

### Optimizers

-   Stochastic Gradient Descent (SGD)\
-   Momentum\
-   Nesterov Accelerated Gradient (NAG)\
-   RMSProp\
-   Adam\
-   Nadam

### Weight Initialization

-   Small random Gaussian initialization\
-   Xavier (Glorot) initialization\
-   Zero initialization (for experimentation)

------------------------------------------------------------------------

##  Training the Model

Example command:

``` bash
python src/train.py -d mnist -e 10 -b 64 -l cross_entropy -o adam --lr 0.001 --nhl 2 --sz 128 64 -a relu --w_i xavier
```

This command: - Trains the model\
- Logs metrics (if enabled)\
- Saves the best-performing model in the `models/` directory

Saved outputs: - `best_model.npy` -- Trained weights\
- `best_config.json` -- Best configuration

------------------------------------------------------------------------

##  Running Inference

To evaluate the saved model:

``` bash
python src/inference.py
```

The script outputs: - Loss\
- Accuracy\
- Precision\
- Recall\
- F1-Score


------------------------------------------------------------------------


##  Objective

The primary goal of this project is to understand neural network
training from first principles by manually implementing:

-   Forward propagation\
-   Backpropagation\
-   Gradient computation\
-   Optimization updates\
-   Model evaluation

This project emphasizes clarity, modular design, and mathematical
correctness.
