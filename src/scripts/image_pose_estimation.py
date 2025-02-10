import os
import cv2
from PIL import Image
from .utils.file import read_from_url_array
from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType

def run() -> None:
    estimator = SapiensPoseEstimation(SapiensPoseEstimationType.POSE_ESTIMATION_1B)

    img = read_from_url_array("https://learnopencv.com/wp-content/uploads/2024/09/football-soccer-scaled.jpg")
    pose_estimation, keypoints = estimator(img)

    try:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        Image.fromarray(pose_estimation).save("outputs/image_pose_estimation.png")
        print("Image saved in outputs.")
    except Exception as e:
        print(f"Error saving image: {e}")

    try:
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.imshow("pose_estimation", pose_estimation)
        cv2.waitKey(0)
    except Exception as e:
        print(f"Error when displaying the image: {e}")
