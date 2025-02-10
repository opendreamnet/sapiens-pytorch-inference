import os
from PIL import Image
from .utils.file import read_from_url_array
from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType

def run() -> None:
    pass
    # config = SapiensConfig()
    # config.depth_type = SapiensDepthType.DEPTH_1B
    # config.normal_type = SapiensNormalType.NORMAL_1B
    # predictor = SapiensPredictor(config)

    # img = read_from_url_array("https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/assets/test2.png?raw=true")

    # # Estimate the maps
    # result = predictor(img)

    # try:
    #     if not os.path.exists("outputs"):
    #         os.makedirs("outputs")

    #     Image.fromarray(result).save("outputs/image_predictor.png")
    #     print("Image saved in outputs.")
    # except Exception as e:
    #     print(f"Error saving image: {e}")
