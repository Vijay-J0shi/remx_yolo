
# Image Processing and Augmentation for Object Detection

This project provides a set of utilities to prepare a dataset of images, with a focus on bounding boxes, for training machine learning models. It includes the functionality to rename, resize, apply letterboxing, transform bounding boxes, and apply augmentations using `albumentations`.

## Requirements

- **Python 3.x**
- **OpenCV**: For image loading and resizing.
- **Numpy**: For array operations.
- **Albumentations**: For image augmentations.

### Install required packages:
```bash
pip install opencv-python numpy albumentations
```

## Functionality

### 1. Image Renaming

The `img_rename()` function renames all image files in a given folder and its subfolders by appending a timestamp and a random string to avoid filename clashes.

### 2. Image Resizing with Letterboxing

The `img_resize()` function resizes images to the specified size and saves them in the output directory. You can choose to maintain the aspect ratio of images using letterboxing.

### 3. Bounding Box Transformation

There are two functions provided to handle bounding box transformations:
- `letterbox_coordinate_transform()`: Transforms bounding boxes to match the resized images with letterboxing.
- `inverse_letterbox_coordinate_transform()`: Transforms the bounding boxes back to their original dimensions.

### 4. Data Augmentation

The `augmentation_transforms()` function provides image augmentations such as flips, rotations, scaling, and brightness adjustments using the `albumentations` library.

## Script Overview

### Key Parameters:

- **input_dir**: Directory containing the raw images.
- **output_dir**: Directory where renamed and resized images will be saved.
- **img_size**: Tuple specifying the target size for image resizing.
- **letter_box**: Boolean flag to decide whether to apply letterboxing or not.

### Steps:

1. **Image Renaming**: Rename all images in the `input_dir` to avoid overwriting by appending the folder name, current timestamp, and a random string.
   ```python
   img_rename(input_dir)
   ```

2. **Image Resizing**: Resize images with the option to maintain the aspect ratio using letterboxing.
   ```python
   img_resize(input_dir, output_dir, (640, 640), letter_box=True)
   ```

3. **Bounding Box Transformation**: Adjust the bounding boxes to the resized images (useful for object detection tasks).
   ```python
   new_bboxes = letterbox_coordinate_transform(bboxes, original_size, letterboxed_size)
   ```

4. **Data Augmentation**: Apply a range of augmentations to images to enhance dataset variety.
   ```python
   transform = augmentation_transforms()
   ```

## Directory Structure:

```bash
.
├── datasets/
│   ├── raw-images/           # Original raw images
│   ├── resized-images/       # Renamed and resized images
├── utils/
│   ├── images.py             # Utility functions for image processing and augmentation
└── README.md                 # This README file
```

## Utility Functions:

### Image Renaming:
```python
def img_rename(folder_path: str)
```
Renames images in the provided folder by appending a unique identifier to avoid overwriting files.

### Image Resizing:
```python
def img_resize(input_dir: str, output_dir: str, img_size: tuple, letter_box: bool = True)
```
Resizes images, with an optional letterboxing feature to maintain the aspect ratio.

### Bounding Box Transformation:
```python
def letterbox_coordinate_transform(bboxes: List[BBox], original_size: ImgSize, letterboxed_size: ImgSize)
def inverse_letterbox_coordinate_transform(bboxes: List[BBox], original_size: ImgSize, letterboxed_size: ImgSize)
```
These functions adjust bounding boxes to fit the resized or original image dimensions.

### Data Augmentation:
```python
def augmentation_transforms()
```
Provides augmentation techniques like horizontal/vertical flips, scaling, and brightness adjustments.

## Example Usage:

1. **Renaming and Resizing Images**:
   ```python
   img_rename("datasets/raw-images")
   img_resize("datasets/raw-images", "datasets/resized-images", (640, 640), letter_box=True)
   ```

2. **Bounding Box Transformation**:
   ```python
   transformed_bboxes = letterbox_coordinate_transform(bboxes, original_size, new_size)
   ```

3. **Apply Augmentations**:
   ```python
   augment = augmentation_transforms()
   augmented_image = augment(image=image)
   ```

## Notes:
# letterbox
- The `letterbox` function maintains aspect ratios by padding images to the target size with a fill value of `114`.
- Bounding box coordinates are transformed to match the new image size and can be inverted back to their original coordinates if needed.
- The augmentation pipeline can be customized by adding/removing transformations based on project requirements.

The **letterbox** technique in image processing is a way to resize an image while preserving its original aspect ratio, and then padding the empty space with a background color (usually black or gray). This is commonly used when you need to resize an image to a fixed size (e.g., 640x640), but you don’t want to distort the image by stretching or squashing it.

Here’s how the **letterbox** technique works in detail:

### Step-by-Step Explanation

1. **Original Image and Target Size**:
   - You start with an original image of any dimensions, say 800x600 (width x height).
   - You want to resize this image to a fixed target size, for example, 640x640 (a square).

Here is the given explanation converted into a `README.md` format:

### Aspect Ratios

The aspect ratio of the original image is calculated as:

```markdown
Aspect Ratio (original) = original width / original height = 800 / 600 = 1.33
```

For the target size of 640x640, the aspect ratio is:

```markdown
Aspect Ratio (target) = 640 / 640 = 1
```

Since the original aspect ratio (1.33) differs from the target (1), directly resizing the image would distort it.

### Determine Scaling Factor

To resize without distortion, we need to scale the image to fit within the target dimensions while maintaining the original aspect ratio. The scaling factor is based on the smaller ratio between the target and original dimensions:

```markdown
Scaling Factor = min(640 / 800, 640 / 600) = min(0.8, 1.066) = 0.8
```

Thus, the image will be resized by 80%, resulting in the following dimensions:

```markdown
New Width = 800 * 0.8 = 640 pixels
New Height = 600 * 0.8 = 480 pixels
```

### Resize the Image

The image is resized to 640x480, which fits within the 640x640 target, but it leaves empty space vertically since the height is smaller than 640 pixels.

### Add Padding

To center the resized image (640x480) within the 640x640 frame, padding needs to be added to the top and bottom:

```markdown
Vertical Padding = (640 - 480) / 2 = 80 pixels
```

Therefore, 80 pixels of padding are added to the top and bottom. The final image size will be 640x640 without distortion. Horizontal padding is not needed, as the width already matches the target.

### Final Image

The final image is 640x640 with the original aspect ratio preserved by adding 80 pixels of padding on both the top and bottom.

- You get a new image of size 640x640 where the original image is centered with padding on the top and bottom, but the aspect ratio of the original image is preserved.
  
### Visual Breakdown:

1. Original Image: 
   - Size: 800x600
2. Target Size: 
   - Size: 640x640
3. Resized Image (640x480) + Padding (80 pixels top/bottom):Letterbox Image
   - Final size: 640x640



### Key Points:

1. **Aspect Ratio Preservation**: The core of the letterbox technique is maintaining the aspect ratio of the original image by resizing it to fit within the target dimensions.
2. **Padding**: Any unused space after resizing is filled with padding (the default value is `114`, a neutral gray).
3. **No Distortion**: Unlike regular resizing that could distort the image, letterboxing ensures that the content of the image remains intact, with padding added only to the empty space.

### Common Use Case:

Letterboxing is often used in:
- **Object Detection**: When models like YOLO or SSD require images of a fixed size but input images can have varying aspect ratios, letterboxing helps maintain the original image's integrity while fitting it into the required dimensions.
Here's a draft of a `README.md` file for your project:

---

# Object Detection with ONNX Model and Letterbox Image Preprocessing

## Overview

This project demonstrates how to perform object detection using a machine learning model in ONNX format. The pipeline processes images, applies letterboxing for resizing, performs inference using the ONNX model, filters predictions, and draws bounding boxes on detected objects. The project is intended for detecting specific classes, such as animals like axis-deer and elephants, but can be adapted for other classes as well.

### Key Features:
- **Letterbox Resizing**: Resizes images without distorting the aspect ratio, adding padding when necessary.
- **Bounding Box Predictions**: Converts model outputs into bounding boxes.
- **Non-Maximum Suppression (NMS)**: Filters overlapping bounding boxes using IoU thresholds.
- **ONNX Inference**: Runs inference using ONNX models.
- **Bounding Box Visualization**: Draws bounding boxes around detected objects with confidence scores.

## Requirements

- Python 3.8+
- [OpenCV](https://opencv.org/) (`cv2`)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Pillow](https://python-pillow.org/) (`PIL`)
- Numpy (`numpy`)

To install the necessary dependencies, use:

```bash
pip install numpy opencv-python onnxruntime pillow
```

## Project Structure

- **images.py**: Contains utility functions for image preprocessing, letterbox resizing, and inverse coordinate mapping.
- **object_detection.py**: Main script that handles ONNX inference, bounding box filtering, and drawing results.

### Key Functions:

1. **`letterbox()`**:
   Resizes the input image to a target size while maintaining the original aspect ratio and adding padding.

2. **`nms()`**:
   Implements Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes based on IoU (Intersection Over Union).

3. **`xywh2xyxy()`**:
   Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.

4. **`final_image_pre_process()`**:
   Prepares the image for ONNX model inference by resizing and normalizing pixel values.

5. **`draw_bboxes()`**:
   Draws bounding boxes on the image after prediction.

6. **`model_ort_session()`**:
   Loads the ONNX model and retrieves input/output names for inference.

7. **`predict_images()`**:
   Main function that processes images, runs model inference, and generates results.

## Usage

1. **Prepare your ONNX model**: Ensure you have a trained ONNX model compatible with your dataset.

2. **Predict Images**:
   Call `predict_images()` with the model path and directory containing images. The results include bounding boxes, confidence scores, and labels.

   ```python
   MODEL = "path/to/your/model.onnx"
   DIRECTORY = "path/to/your/image/directory"

   results = predict_images(MODEL, DIRECTORY)
   ```

3. **View Results**: Each image will be saved with detected bounding boxes drawn. You can visualize the results directly from the returned images.

## Example

```python
from object_detection import predict_images

MODEL_PATH = "model.onnx"
IMAGE_DIR = "data/images"

results = predict_images(MODEL_PATH, IMAGE_DIR)

# Save or display results as needed
for label, predictions in results.items():
    for image_name, prediction in predictions.items():
        output_img = prediction['pred_img']
        cv2.imshow("Detected Image", output_img)
        cv2.waitKey(0)  # Press any key to close the image window
```
