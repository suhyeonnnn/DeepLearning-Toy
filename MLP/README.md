# MLP Image Classifier (Fashion-MNIST)

A simple image classification project using Multi-Layer Perceptron (MLP) on the Fashion-MNIST dataset.

## Overview
- Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Model: Input → Flatten → Dense(128, ReLU) × 2 → Dense(10, Softmax)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

## Files
- `main.py`: Entry point for loading data, training, and prediction
- `ImageClassifier_MLP.py`: MLP model class definition
- `onehot_utils.py`: One-hot encoding utility function

## Result
| Sample Output |
|---------------|
| ![prediction](results/sample_prediction.png) |

## How to Run
```bash
pip install -r requirements.txt
python main.py

