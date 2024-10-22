
# Image Processing and Dataset Splitting for Animal Images

This repository contains a script for renaming, resizing, and splitting a dataset of animal images into training, validation, and test sets. The code utilizes utility functions and `splitfolders` to facilitate dataset preparation for machine learning tasks such as object detection or image classification.

## Requirements

- **Python 3.x**
- **split-folders**: A Python package to split datasets into train, validation, and test sets.
- **Custom utility functions** for image renaming and resizing, assumed to be in `utils/images.py`.

### Install required packages:
```bash
pip install split-folders
```

## Functionality

### 1. Image Renaming

The `img_rename()` function renames images in the input directory.

### 2. Image Resizing

The `img_resize()` function resizes all images to a specified size and saves them in the output directory.

### 3. Dataset Splitting

The `splitfolders.ratio()` function splits the dataset into training, validation, and test sets based on the specified ratio.

## Script Overview

### Key Parameters:

- **input_dir**: Directory containing the raw animal images.
- **non_split_dir**: Directory where renamed and resized images will be saved before splitting.
- **animals_dir**: Output directory for the final split dataset.

### Steps:

1. **Image Renaming**: The script renames all images in the `input_dir` directory to ensure consistency.
   ```python
   img_rename(input_dir)
   ```

2. **Image Resizing**: The resized images are saved in `non_split_dir`. The target size for the images is `(640, 640)` in this example.
   ```python
   img_resize(input_dir, non_split_dir, (640, 640))
   ```

3. **Dataset Splitting**: The renamed and resized images are split into training, validation, and test sets with a ratio of 90% training, 5% validation, and 5% testing.
   ```python
   splitfolders.ratio(
       non_split_dir,
       output=animals_dir,
       seed=1337,
       ratio=(0.9, 0.05, 0.05),
       move=False,
   )
   ```

## Directory Structure:

```bash
.
├── datasets/
│   ├── raw-animals-img/      # Original raw images of animals
│   ├── animals-nosplit/      # Renamed and resized images before splitting
│   └── animals/              # Final split dataset
│       ├── train/            # Training images
│       ├── val/              # Validation images
│       └── test/             # Test images
└── utils/
    └── images.py             # Utility functions for image renaming and resizing
```

### Custom Utility Functions (`utils/images.py`):

- `img_rename(input_dir)`: Renames all the images in the input directory.
- `img_resize(input_dir, output_dir, target_size)`: Resizes images to the target size and saves them in the output directory.

## Notes:

- **Splitting Ratio**: The current ratio of 90% training, 5% validation, and 5% testing can be adjusted as needed by modifying the `ratio` parameter.
- **move**: Set to `False` in the `splitfolders.ratio()` function to copy the images to the new folders instead of moving them. Set to `True` if you prefer to move the images.

## How to Run

1. Clone this repository.
2. Place your raw images in the `datasets/raw-animals-img/` directory.
3. Run the script to rename, resize, and split the dataset.

---


# YOLOv8 Transfer Learning for Object Detection

This repository demonstrates how to perform transfer learning using a pre-trained YOLOv8 model. The model is fine-tuned for a custom object detection task and trained on a new dataset.

## Requirements

- **Python 3.x**
- **PyTorch**
- **Ultralytics YOLO**
- **CUDA (optional for GPU training)**

### Install required packages:
```bash
pip install ultralytics torch
```

## Model Training and Fine-tuning

The script utilizes transfer learning to fine-tune a YOLOv8 model on a custom dataset.

### Key Features:

- **Transfer Learning**: Fine-tune a pre-trained YOLOv8 model on your dataset.
- **Freezing Layers**: The ability to freeze specific layers of the model during training.
- **Custom Dataset**: Define your dataset in a `config.yaml` file.
- **ONNX Export**: Export the trained model in ONNX format for further usage.

### Steps to Train the Model:

1. **Model Setup**:
   The pre-trained YOLOv8 model is loaded using:
   ```python
   model = YOLO(model="yolov8m.pt")
   ```

2. **Training**:
   The model is fine-tuned on a custom dataset specified in `config.yaml`.
   ```python
   model.train(
       data="path_to_your_dataset/config.yaml",
       task="detect",
       batch=1,
       device=device,  # GPU if available, otherwise CPU
       imgsz=640,
       epochs=15,
       seed=14,
   )
   ```

3. **Model Freezing** (optional):
   You can freeze specific layers of the model to avoid retraining them by uncommenting and using the `freeze_layers_fn`.

### Configuring the Dataset:

Ensure that your dataset is properly formatted and specified in a `config.yaml` file.

Example `config.yaml`:
```yaml
train: path/to/train/images
val: path/to/val/images
nc: 2  # Number of classes in the dataset
names: ['class1', 'class2']  # Class names
```

### Predicting with the Model:

After training, you can make predictions on test images:
```python
results_detection = model.predict(
    source="path_to_test_images",
    model="path_to_trained_weights/best.pt",
)
```

### Exporting the Model:

Once training is complete, you can export the trained model in ONNX format for deployment:
```python
path = model.export(format="onnx")
```

## Directory Structure:

```
.
├── datasets/
│   ├── config.yaml          # Dataset configuration file
│   └── train/               # Training images
│   └── val/                 # Validation images
│   └── test/                # Test images
├── models/
│   └── yolov8m.pt           # Pre-trained YOLOv8 model
├── results/
│   └── detect/              # Directory for saving detection results
└── remx_transfer_learning.ipynb             # Script for model training
```

## Notes:

- **Pre-trained Model**: The pre-trained YOLOv8 model (`yolov8m.pt`) is loaded from the `Ultralytics` repository. You can change this to other available versions (`yolov8n`, `yolov8s`, etc.).
- **Model Freezing**: Layer freezing is an optional step to reduce the computational cost during training. You can selectively freeze layers to prevent them from updating during training.
- **Batch Size**: Adjust the batch size based on the computational resources available.
  
---

### References:

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
