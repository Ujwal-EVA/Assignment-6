
# PyTorch CNN for MNIST Classification

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits from the MNIST dataset. The architecture includes multiple convolutional layers, batch normalization, dropout for regularization, and global average pooling for efficient feature reduction.

## Features
- **Deep CNN Architecture**: 
  - Multiple convolutional layers with batch normalization and ReLU activation.
  - Dropout layers to prevent overfitting.
  - Global Average Pooling (GAP) to reduce feature dimensions.
- **Dataset**: MNIST (handwritten digit classification with 10 classes).
- **Training and Testing**: Implements functions for model training and evaluation.
- **Visualization**: Progress bars for training using `tqdm`& `matplotlib`

## Requirements
Make sure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- torchsummary
- matplotlib

## Usage

### Training and Testing the Model
1. Run the Python script to train and test the model:
   ```bash
   python model.py
   ```

2. The model will:
   - Train for 20 epochs (default setting).
   - Evaluate on the test set after each epoch.
   - Output the training loss and accuracy for each epoch.

### Model Architecture
The CNN consists of:
- **Convolutional Layers**: Extract spatial features from input images 1 --> 8 --> 16 -->32.
- **Batch Normalization**: Normalizes intermediate feature maps to speed up training.
- **Max Pooling**: Reduces the size by half
- **Dropout**: Prevents overfitting by randomly deactivating neurons during training by 10%.
- **Global Average Pooling**: Reduces feature maps to a single value (1 x 1) per channel.
- **Fully Connected Layer**: Outputs probabilities for 10 classes.

### Example Output
A typical training log might look like:
```
loss=0.032 batch_id=100
Test set: Average loss: 0.0150, Accuracy: 9850/10000 (98%)
```

## GitHub Actions
The repository includes a GitHub Actions workflow for Continuous Integration (CI). The workflow:
- Installs dependencies.
- Trains the model for a few epochs.
- Verifies the implementation by testing on the MNIST dataset.

### Running the Workflow
1. Push changes to the `main` branch.
2. View the workflow logs in the GitHub Actions tab.

