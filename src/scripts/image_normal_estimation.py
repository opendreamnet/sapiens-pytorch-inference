import cv2
import os
from .utils.file import read_from_url_array
from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

def run() -> None:
    estimator = SapiensNormal(SapiensNormalType.NORMAL_1B)

    img = read_from_url_array("https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/assets/test1.png?raw=true")

    normal_map = estimator(img)
    normal_map = draw_normal_map(normal_map)
    combined = cv2.addWeighted(img, 0.3, normal_map, 0.8, 0)

    try:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        cv2.imwrite("outputs/image_normal_estimation.jpg", combined)
        print("Image saved in outputs.")
    except Exception as e:
        print(f"Error saving image: {e}")

    try:
        cv2.namedWindow("Normal Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Normal Map", combined)
        cv2.waitKey(0)
    except Exception as e:
        print(f"Error when displaying the image: {e}")

