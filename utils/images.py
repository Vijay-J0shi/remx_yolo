import os
import cv2
import numpy as np

# import sys  # for cli implementation


# def img_rename(folder_path: str):
#     """
#     The `img_rename` function renames all image files in a given folder and its subfolders by appending
#     the parent folder name and a unique index to the original file name.

#     :param folder_path: The `folder_path` parameter is a string that represents the path to the folder
#     where the image files are located
#     :type folder_path: str
#     """
#     for root, dirnames, filenames in os.walk(folder_path):
#         for dirname in dirnames:
#             i = 1  # Start the index at 1
#             nested_folder_path = os.path.join(root, dirname)
#             for filename in os.listdir(nested_folder_path):
#                 if filename.endswith(".jpg") or filename.endswith(".png"):
#                     base_filename, file_extension = os.path.splitext(filename)
#                     new_filename = f"{dirname}-{i}{file_extension}"
#                     i += 1
#                     os.rename(
#                         os.path.join(nested_folder_path, filename),
#                         os.path.join(nested_folder_path, new_filename),
#                     )


# Resolve files overwritten or disappearing.
# The new filenames clash and result in overwriting of old of same base filename in subfolder.
def generate_random_string(length=6):
    """
    The function `generate_random_string` generates a random string of lowercase letters with a
    specified length.

    :param length: The `length` parameter is an optional parameter that specifies the length of the
    random string to be generated. By default, if no value is provided for `length`, it will be set to
    6, defaults to 6 (optional)
    :return: The function `generate_random_string` returns a randomly generated string of lowercase
    letters with a length specified by the `length` parameter.
    """
    import random
    import string

    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def img_rename(folder_path: str):
    """
    The `img_rename` function renames all image files in a given folder and its subfolders by appending
    the folder name, current timestamp, and a random string to the original file name.

    :param folder_path: The `folder_path` parameter is a string that represents the path to the folder
    where the image files are located
    :type folder_path: str
    """
    import datetime

    for root, dirnames, filenames in os.walk(folder_path):
        for dirname in dirnames:
            i = 1
            nested_folder_path = os.path.join(root, dirname)
            for filename in os.listdir(nested_folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    base_filename, file_extension = os.path.splitext(filename)
                    random_string = generate_random_string(4)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    new_filename = (
                        f"{dirname}-{timestamp}-{random_string}{file_extension}"
                    )
                    os.rename(
                        os.path.join(nested_folder_path, filename),
                        os.path.join(nested_folder_path, new_filename),
                    )


# Example usage
img_rename("/path/to/your/folder")


# Example usage:
img_rename("/path/to/your/folder")


class ImgSize:
    def __init__(self, width: int, height: int, channel: int = 3) -> None:
        self.height = height
        self.width = width
        self.channel = channel

    def get_tuple(self) -> tuple:
        return (self.width, self.height, self.channel)


def letterbox(img: np.ndarray, new_size: ImgSize, fill_value: int = 114) -> np.ndarray:
    # [why fill_value = 114](https://github.com/ultralytics/ultralytics/blob/796bac229eb5040159d7dff549f136f8c7e1c64e/ultralytics/data/augment.py#L587)
    aspect_ratio = min(new_size.height / img.shape[1], new_size.width / img.shape[0])

    new_size_with_ar = int(img.shape[1] * aspect_ratio), int(
        img.shape[0] * aspect_ratio
    )

    resized_img = np.asarray(cv2.resize(img, new_size_with_ar))
    resized_h, resized_w, _ = resized_img.shape

    padded_img = np.full(new_size.get_tuple(), fill_value)
    center_x = new_size.width / 2
    center_y = new_size.height / 2

    x_range_start = int(center_x - (resized_w / 2))
    x_range_end = int(center_x + (resized_w / 2))

    y_range_start = int(center_y - (resized_h / 2))
    y_range_end = int(center_y + (resized_h / 2))

    padded_img[y_range_start:y_range_end, x_range_start:x_range_end, :] = resized_img
    return padded_img


def img_resize(
    input_dir: str, output_dir: str, img_size: tuple, letter_box: bool = True
):
    """
    The `img_resize` function resizes images in a given input directory to a specified size and saves
    them in an output directory, with an option to letterbox the images.

    :param input_dir: The input directory where the original images are located
    :type input_dir: str
    :param output_dir: The output directory where the resized images will be saved
    :type output_dir: str
    :param img_size: The `img_size` parameter is a tuple that specifies the desired size of the resized
    image. It should be in the format `(width, height)`, where `width` and `height` are integers
    representing the width and height in pixels, respectively
    :type img_size: tuple
    :param letterbox: The `letterbox` parameter is a boolean flag that determines whether the images
    should be resized using letterboxing or not. If `letterbox` is set to `True`, the images will be
    resized while maintaining their aspect ratio by adding black bars to the sides or top/bottom of the
    image, defaults to True
    :type letterbox: bool (optional)
    """
    # Check if the directory exists
    if not os.path.exists(output_dir):
        # Create the directory if it does not exist
        os.makedirs(output_dir)

    # Loop through all subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Loop through all files in the subdirectory
        for filename in files:
            # Check if the file is an image
            if (
                filename.endswith(".jpg")
                or filename.endswith(".jpeg")
                or filename.endswith(".png")
            ):
                # Load the image
                img = cv2.imread(os.path.join(root, filename))

                if letter_box:
                    resized_img = letterbox(
                        np.asarray(img), ImgSize(img_size[0], img_size[1])
                    )
                else:
                    # Resize the image to 640x640
                    resized_img = cv2.resize(img, img_size)

                # Save the resized image to the output directory
                output_subdir = os.path.join(
                    output_dir, os.path.relpath(root, input_dir)
                )
                os.makedirs(output_subdir, exist_ok=True)
                cv2.imwrite(os.path.join(output_subdir, filename), resized_img)


from typing import List, Tuple

# type alias BBox as a tuple of four floats representing
# the coordinates of a bounding box in the format (x1, y1, x2, y2),
# where (x1, y1) is the top-left corner and (x2, y2) bottom-right corener of single bounding box
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2) for single bounding box


def inverse_letterbox_coordinate_transform(
    bboxes: List[BBox], original_size: ImgSize, letterboxed_size: ImgSize
) -> List[BBox]:
    """
    The `inverse_letterbox_coordinate_transform` function takes a list of bounding boxes, the original
    image size, and the letterboxed image size, and returns the bounding boxes transformed back to the
    original image dimensions.

    :param bboxes: The `bboxes` parameter is a list of bounding boxes. Each bounding box is represented
    as a tuple of four values: `(x1, y1, x2, y2)`. `x1` and `y1` are the coordinates of the top-left
    corner of the bounding box
    :type bboxes: List[BBox]
    :param original_size: The original_size parameter represents the dimensions of the original image
    before it was letterboxed. It is an ImgSize object that contains the width and height of the
    original image
    :type original_size: ImgSize
    :param letterboxed_size: The `letterboxed_size` parameter represents the size of the image after it
    has been letterboxed. It is an `ImgSize` object that contains the width and height of the
    letterboxed image
    :type letterboxed_size: ImgSize
    :return: a list of bounding boxes in the original image dimensions.
    """

    # Calculate the aspect ratio of the original and letterboxed sizes
    aspect_ratio = min(
        letterboxed_size.height / original_size.width,
        letterboxed_size.width / original_size.height,
    )

    # Calculate the amount of padding added during the letterbox operation
    pad_w = letterboxed_size.width - (aspect_ratio * original_size.width)
    pad_h = letterboxed_size.height - (aspect_ratio * original_size.height)

    # Convert the bounding box coordinates back to the original image dimensions
    inverse_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # TODO(Adam-Al-Rahman): Better method than `round`
        # (x1, y1) is the top-left corner of single bounding box
        map_x1 = round(x1 / aspect_ratio - pad_w / (2 * aspect_ratio))
        map_y1 = round(y1 / aspect_ratio - pad_h / (2 * aspect_ratio))

        # (x2, y2) is the bottom-right corner of single bounding box
        map_x2 = round(x2 / aspect_ratio - pad_w / (2 * aspect_ratio))
        map_y2 = round(y2 / aspect_ratio - pad_h / (2 * aspect_ratio))
        inverse_bboxes.append((map_x1, map_y1, map_x2, map_y2))
    return inverse_bboxes


def augmentation_transforms():
    import albumentations as A

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomScale(scale_limit=0.2, interpolation=cv2.INTER_LINEAR, p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=(
                    114,
                    114,
                    114,
                ),  # (https://github.com/ultralytics/ultralytics/blob/796bac229eb5040159d7dff549f136f8c7e1c64e/ultralytics/data/augment.py#L587)
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ],
                p=0.2,
            ),
            # TODO(Adam-Al-Rahman): Check if is required else remove
            # A.OneOf(
            #     [
            #         A.RandomRain(blur_value=3, brightness_coefficient=0.8, p=0.1),
            #         A.RandomSnow(
            #             snow_point_lower=0.1,
            #             snow_point_upper=0.3,
            #             brightness_coeff=2,
            #             p=0.1,
            #         ),
            #     ],
            #     p=0.2,
            # ),
            # A.OneOf(
            #     [
            #         A.GaussNoise(var_limit=(10, 50)),
            #         A.MultiplicativeNoise(),
            #         A.ImageCompression(quality_lower=50, quality_upper=99),
            #     ],
            #     p=0.2,
            # ),
            # A.OneOf(
            #     [
            #         A.ElasticTransform(
            #             alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            #         ),
            #         A.GridDistortion(),
            #         A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            #     ],
            #     p=0.3,
            # ),
        ]
    )
    return transform


# TODO(Adam-Al-Rahman): Implement the cli of the above functions
# CLI usage

# if len(sys.argv) < 3:
#     print("Usage: python script.py folder_path ")
# else:
#     folder_path = sys.argv[1]
#     print("Folder path:", folder_path)
