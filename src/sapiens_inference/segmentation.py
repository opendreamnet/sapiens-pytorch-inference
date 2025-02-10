from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .common import create_preprocessor, download_hf_model

class SapiensSegmentationType(Enum):
    # Enum for different segmentation model types with their respective file paths
    SEGMENTATION_03B = "sapiens-seg-0.3b-torchscript/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
    SEGMENTATION_06B = "sapiens-seg-0.6b-torchscript/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
    SEGMENTATION_1B = "sapiens-seg-1b-torchscript/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"

# Initialize random state for reproducibility
# random = np.random.RandomState(11)

# List of class names for segmentation
# classes = [
#     "Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand",
#     "Left Lower Arm", "Left Lower Leg", "Left Shoe", "Left Sock",
#     "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot",
#     "Right Hand", "Right Lower Arm", "Right Lower Leg", "Right Shoe",
#     "Right Sock", "Right Upper Arm", "Right Upper Leg", "Torso",
#     "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth",
#     "Upper Teeth", "Tongue"
# ]

# # Generate random colors for each class
# colors = random.randint(0, 255, (len(classes) - 1, 3))
# colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
# colors = colors[:, ::-1]  # Convert BGR to RGB

ORIGINAL_GOLIATH_CLASSES = (
    "Background",
    "Apparel",
    "Chair",
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Face_Neck",
    "Hair",
    "Headset",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Lower_Spandex",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Visible_Badge",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

ORIGINAL_GOLIATH_PALETTE = [
    [50, 50, 50],
    [255, 218, 0],
    [102, 204, 0],
    [14, 0, 204],
    [0, 204, 160],
    [128, 200, 255],
    [255, 0, 109],
    [0, 255, 36],
    [189, 0, 204],
    [255, 0, 218],
    [0, 160, 204],
    [0, 255, 145],
    [204, 0, 131],
    [182, 0, 255],
    [255, 109, 0],
    [0, 255, 255],
    [72, 0, 255],
    [204, 43, 0],
    [204, 131, 0],
    [255, 0, 0],
    [72, 255, 0],
    [189, 204, 0],
    [182, 255, 0],
    [102, 0, 204],
    [32, 72, 204],
    [0, 145, 255],
    [14, 204, 0],
    [0, 128, 72],
    [204, 0, 43],
    [235, 205, 119],
    [115, 227, 112],
    [157, 113, 143],
    [132, 93, 50],
    [82, 21, 114],
]

## 6 classes to remove
REMOVE_CLASSES = (
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Visible_Badge",
    "Chair",
    "Lower_Spandex",
    "Headset",
)

## 34 - 6 = 28 classes left
GOLIATH_CLASSES = tuple(
    [x for x in ORIGINAL_GOLIATH_CLASSES if x not in REMOVE_CLASSES]
)

GOLIATH_PALETTE = [
    ORIGINAL_GOLIATH_PALETTE[idx]
    for idx in range(len(ORIGINAL_GOLIATH_CLASSES))
    if ORIGINAL_GOLIATH_CLASSES[idx] not in REMOVE_CLASSES
]

# def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
#     """Draws the segmentation map using the predefined colors."""
#     h, w = segmentation_map.shape
#     segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)

#     # Map each class index to its corresponding color
#     for i, color in enumerate(colors):
#         segmentation_img[segmentation_map == i] = color

#     return segmentation_img

def postprocess_segmentation(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    """Postprocess the model results to generate the segmentation mask."""
    # Resize the output to match the original image shape
    results = F.interpolate(results, size=img_shape, mode="bilinear", align_corners=False)
    _, preds = torch.max(results, 1)  # Get the predicted class indices
    return preds.squeeze(0).cpu().numpy()

def visualize_pred_with_overlay(img: Image.Image, sem_seg, alpha=0.5):
    img_np = np.array(img.convert("RGB"))
    sem_seg = np.array(sem_seg)

    num_classes = len(GOLIATH_CLASSES)
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [GOLIATH_PALETTE[label] for label in labels]

    overlay = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)

    for label, color in zip(labels, colors):
        overlay[sem_seg == label, :] = color

    blended = np.uint8(img_np * (1 - alpha) + overlay * alpha)
    return Image.fromarray(blended)

class SapiensSegmentation:
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32,
                 cache_dir = None):
        # Download and load the specified model
        path = download_hf_model(type.value, cache_dir=cache_dir)
        self.model = torch.jit.load(path).eval().to(device).to(dtype)  # Load model and set to evaluation mode
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768))  # Preprocessor for input image resizing

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Runs the segmentation model on the input image."""
        tensor = self.preprocessor(img).unsqueeze(0).to(self.device).to(self.dtype)  # Preprocess and convert to tensor

        with torch.inference_mode():  # Disable gradient calculation for inference
            results = self.model(tensor)  # Run the model
        return postprocess_segmentation(results, img.shape[:2])  # Postprocess results to get segmentation mask
