# byop_shivam

Model 1

### Main Model Structure
The neural network architecture includes:
- 4 convolutional blocks with increasing channel dimensions (64 → 128 → 256 → 512)
- Each block contains:
  - 2D Convolution with 3x3 kernel
  - Batch Normalization
  - ReLU activation
  - Max Pooling
- Global Attention layer after the convolutional blocks
- Fully connected layers with dropout for regularization
- Output layer designed for multi-character prediction

### Dataset Preparation

The model uses a custom dataset class that expects:
- A list of image paths
- One-hot encoded ground truth labels
- Optional transforms

```python
dataset = CustomDataset(
    image_paths=your_image_paths,
    gt_one_hot=your_labels,
    transform=your_transforms
)
```

Model 2
# Multi-Output CNN Character Recognition Model

## Overview
This model is a Convolutional Neural Network (CNN) designed for character recognition tasks. It features a multi-output architecture capable of predicting 5 characters simultaneously, making it suitable for tasks like CAPTCHA recognition or multi-character sequence detection.

### Convolutional Layers
1. First Convolutional Block
   - Conv2D: 16 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D
   
2. Second Convolutional Block
   - Conv2D: 32 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D
   
3. Third Convolutional Block
   - Conv2D: 32 filters, 3x3 kernel, ReLU activation
   - BatchNormalization
   - MaxPooling2D

### Dense Layers
For each of the 5 outputs:
- Flatten layer (shared)
- Dense layer: 64 units, ReLU activation
- Dropout layer: 0.5 rate
- Output layer: `nchar`(36)units, sigmoid activation

## Model Configuration
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy (tracked for each output)

## Model Features
- Parallel processing of multiple characters
- Dropout layers for regularization
- Batch normalization for stable training
- Shared convolutional features across all outputs

## Dependencies
- TensorFlow/Keras
- Required packages: `tensorflow`, `numpy`

## Notes
- The model expects 50*200 image.
- Each output predicts one character independently and loss is calculated for each character. 
- The architecture is designed to handle fixed-length character sequences
