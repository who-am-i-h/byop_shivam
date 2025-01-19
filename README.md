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
