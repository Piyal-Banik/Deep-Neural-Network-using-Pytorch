# Simple Artificial Neural Network (ANN) with PyTorch

This repository contains an implementation of a simple Artificial Neural Network (ANN) trained on the MNIST dataset using PyTorch. The model is trained using **Stochastic Gradient Descent (SGD)** and evaluated on the test dataset.

## Features

✅ Uses PyTorch’s `nn.Module` to define a fully connected ANN.  
✅ Trains on the **MNIST dataset** (handwritten digits classification).  
✅ Implements a **training loop** with **CrossEntropyLoss** and **SGD optimizer**.  
✅ Includes utilities for **saving and loading models**.  
✅ Supports **GPU acceleration** if available.  

---

## Installation

Ensure you have Python and the required dependencies installed. You can install the necessary libraries using:

```bash
pip install torch torchvision matplotlib
```

## 📁 Project Structure

```
📂 project_root/
│── 📂 data/                # Downloaded dataset (MNIST)
│── 📂 model/               # Model repository
│── 📄 data_setup.py        # Data loading and transformation
│── 📄 model_setup.py       # ANN model definition
│── 📄 training.py          # Training and evaluation script
│── 📄 utils.py             # Model saving utility
│── 📄 models/              # Saved models (after training)
│── 📄 README.md            # Project documentation
```

---

## Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Piyal-Banik/Deep-Neural-Network-using-Pytorch.git
cd your-repository
```

### 2️⃣ Train the Model
Run the `training.py` script to start training the ANN:
```bash
python training.py
```

### 3️⃣ Evaluate the Model
After training, the test accuracy will be displayed in the terminal.

### 4️⃣ Save and Load Model
The trained model is automatically saved in the `models/` directory and can be loaded later for inference.

## Code Breakdown

### 1️⃣ Data Preparation (data_setup.py)
- Loads the MNIST dataset using torchvision.datasets.
- Applies transformations: converts images to tensors and normalizes them. 
- Creates train and test dataloaders.

### 2️⃣ Model Architecture (model_setup.py)

Defines a simple 3-layer ANN using PyTorch's nn.Module:
- Input Layer: Flattens 28×28 images into a 784-dimensional vector.
- Hidden Layer 1: 128 neurons + ReLU activation.
- Hidden Layer 2: 64 neurons + ReLU activation.
- Output Layer: 10 neurons (for digits 0-9).

### 3️⃣ Training Process (training.py)
- Defines loss function (CrossEntropyLoss) and SGD optimizer.
- Trains for 5 epochs, updating weights using backpropagation.
- Evaluates accuracy on the test set.

### 4️⃣ Model Saving (utils.py)
- Saves model weights using torch.save().
- Allows reloading using torch.load().