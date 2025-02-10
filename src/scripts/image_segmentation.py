import os
import numpy as np
from .utils.file import read_from_url
from .utils.image import resize_fill, resize_fill_restore
from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, visualize_pred_with_overlay

def run() -> None:
    estimator = SapiensSegmentation(SapiensSegmentationType.SEGMENTATION_1B)

    img = read_from_url("https://upload.wikimedia.org/wikipedia/commons/5/5b/Jogging_with_dog_at_Carcavelos_Beach.jpg")

    resized_data = resize_fill(img, 1024, 768)
    resized_img = resized_data["resized_image"]

    segmentation_map = estimator(np.array(resized_img))
    segmentation_image = visualize_pred_with_overlay(resized_img, segmentation_map)
    segmentation_image = resize_fill_restore(resized_data, segmentation_image)

    try:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        segmentation_image.save("outputs/image_segmentation.jpg")
        print("Image saved in outputs.")
    except Exception as e:
        print(f"Error saving image: {e}")
