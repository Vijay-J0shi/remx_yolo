import os
import cv2

from PIL import Image
import numpy as np

from images import letterbox, ImgSize, inverse_letterbox_coordinate_transform


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression (NMS)
    Select best bounding box out of a set of overlapping boxes.
    """

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def xywh2xyxy(x):
    """
    yolov8 provide bounding box (x, y, w, h).
    Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_bboxes(img, inverse_coordinates, indices, scores, class_ids, CLASSES):
    img = cv2.imread(img)

    original_img_boxes = []
    new_scores = []
    labels = []  # NOTE: No classification

    for bbox, score, label in zip(
        inverse_coordinates, scores[indices], class_ids[indices]
    ):
        original_img_boxes.append(bbox)
        new_scores.append(score)
        labels.append(label)

        cls_id = int(label)
        cls = CLASSES[cls_id]
        box_color = (255, 0, 255)

        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), box_color, 10)
        cv2.putText(
            img,
            f"{cls}:{int(score*100)}",
            (bbox[0], bbox[1] - 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3.0,
            color=(0, 0, 0),
            thickness=3,
        )

    return {
        "img": img,
        "boxes": original_img_boxes,
        "labels": labels,
    }


def model_ort_session(MODEL: str):
    import onnxruntime as ort

    ort_session = ort.InferenceSession(MODEL)

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]

    return {
        "session": ort_session,
        "input_names": input_names,
        "input_shape": input_shape,
        "output_names": output_names,
    }


def final_image_pre_process(img, input_shape):
    img = cv2.imread(img)  # original image

    # Converting original image into 640x640 size without losing its aspect ratio
    img_letterboxed = letterbox(np.asarray(img), ImgSize(640, 640))

    # TODO(Adam-Al-Rahman): Convert any image format to jpg
    # Convert the np.ndarray to a byte stream
    img_bytes = cv2.imencode(".jpg", img_letterboxed)[1].tobytes()

    # read the image from the byte stream
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[:2]

    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_height, input_width = input_shape[2:]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

    return {
        "input_tensor": input_tensor,
        "image_height": image_height,
        "image_width": image_width,
        "input_height": input_height,
        "input_width": input_width,
    }


def bboxs_filter(outputs, input_width, input_height, image_width, image_height):
    # Threshold
    predictions = np.squeeze(outputs).T
    conf_thresold = 0.85  # confidence score [testing phase]

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    # rescale box
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)

    return {"scores": scores, "boxes": boxes, "class_ids": class_ids}


def map_lb_original_img(original_img, letterboxed_boxes):
    img = cv2.imread(original_img)  # original image
    inverse_coordinates = inverse_letterbox_coordinate_transform(
        # [(x1, y1, x2, y2)]
        letterboxed_boxes,
        ImgSize(img.shape[1], img.shape[0]),
        ImgSize(640, 640),
    )

    return inverse_coordinates


def letterboxed_result(boxes, indices, scores, class_ids, CLASSES=None):
    letterboxed_boxes = []
    new_scores = []
    labels = []

    for bbox, score, label in zip(
        xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]
    ):
        bbox = bbox.round().astype(np.int32).tolist()
        letterboxed_boxes.append(tuple(bbox))
        new_scores.append(score)  # <-- Append to the new list

    return {
        "letterboxed_boxes": letterboxed_boxes,
        "labels": labels,
        "scores": new_scores,
    }


def predict_images(
    MODEL: str,
    main_dir: str,
    img: str = None,
) -> np.ndarray:
    # Check if the directory exists
    if not os.path.exists(main_dir):
        print(f"Error: {main_dir or img} doesn't exist")
        return None

    if img:
        # img = cv2.imread(os.path.join(nested_folder_path, filename))
        # img_letterboxed = final_image_pre_process(img)
        pass

        # TODO(Adam-Al-Rahman): Decide and arrange the session in a way that handle one image followed by folder while testing
        # output = ort_session(img)
    else:
        labels_dir_predict = {}
        for root, labels, _ in os.walk(main_dir):
            for label in labels:
                nested_folder_path = os.path.join(root, label)
                all_file_predict = {}
                for image in os.listdir(nested_folder_path):
                    if image.endswith(".jpg") or image.endswith(".png"):
                        img = os.path.join(nested_folder_path, image)

                        model = model_ort_session(MODEL)
                        pre_process_image = final_image_pre_process(
                            img, model["input_shape"]
                        )

                        outputs = model["session"].run(
                            model["output_names"],
                            {
                                model["input_names"][0]: pre_process_image[
                                    "input_tensor"
                                ]
                            },
                        )[0]

                        bboxs_outputs = bboxs_filter(
                            outputs,
                            pre_process_image["input_width"],
                            pre_process_image["input_height"],
                            pre_process_image["image_width"],
                            pre_process_image["image_height"],
                        )

                        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
                        indices = nms(
                            bboxs_outputs["boxes"], bboxs_outputs["scores"], 0.3
                        )

                        letterboxed_output = letterboxed_result(
                            boxes=bboxs_outputs["boxes"],
                            indices=indices,
                            scores=bboxs_outputs["scores"],
                            class_ids=bboxs_outputs["class_ids"],
                        )

                        inverse_coordinate = map_lb_original_img(
                            img, letterboxed_output["letterboxed_boxes"]
                        )

                        original_image_predict = draw_bboxes(
                            img,
                            inverse_coordinate,
                            indices=indices,
                            scores=bboxs_outputs["scores"],
                            class_ids=bboxs_outputs["class_ids"],
                            CLASSES=["axis-deer", "elephant"],
                        )

                        all_file_predict[root + "/" + label + "/" + image] = {
                            "img": original_image_predict["img"],
                            "coordiante": inverse_coordinate,
                            "labels": label,
                        }
                labels_dir_predict[label] = all_file_predict
        return labels_dir_predict
