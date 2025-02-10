# Sapiens Pytorch Inference
Minimal code and examples for inferencing Sapiens foundation human models in Pytorch

![ONNX Sapiens_normal_segmentation](https://github.com/user-attachments/assets/a8f433f0-5f43-4797-89c6-5b33c58cbd01)

# Why
- Make it easy to run the models by creating a `SapiensPredictor` class that allows to run multiple tasks simultaneously
- Add several examples to run the models on images, videos, and with a webcam in real-time.
- Download models automatically from HuggigFace if not available locally.
- Add a script for ONNX export. However, ONNX inference is not recommended due to the slow speed.
- Added Object Detection to allow the model to be run for each detected person. However, this mode is disabled as it produces the worst results.

> [!CAUTION]
> - Use 1B models, since the accuracy of lower models is not good (especially for segmentation)
> - Exported ONNX models are too slow.
> - Input sizes other than 768x1024 don't produce good results.
> - Running Sapiens models on a cropped person produces worse results, even if you crop a wider rectangle around the person.

## Installation

```bash
pip install git+https://github.com/opendreamnet/sapiens-pytorch-inference.git
```

or

```bash
uv add git+https://github.com/opendreamnet/sapiens-pytorch-inference
```

## Development environment

Clone this repository:

```bash
git clone https://github.com/opendreamnet/sapiens-pytorch-inference.git
```

### Dev Container

> Requirements:
> - [Docker](https://www.docker.com/)
> - [VS Code](https://code.visualstudio.com/)
> - [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

- Open the source folder with VS Code: `code sapiens-pytorch-inference`
- Run the command: `DevContainer: Reopen in Container`
- Wait for the development environment to be prepared.

### Or Manually

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/0.5.24/install.sh | sh`
- Install Python and synchronize dependencies:

```bash
uv python install 3.11
uv venv
uv sync --group scripts
```

### Run examples

**Normal Estimation**: 

`uv run image_normal_estimation`

**Pose Estimation**:

`uv run image_normal_estimation`

**Human Part Segmentation**:

`uv run image_segmentation`

**Export model to ONNX**:

`uv run onnx_export seg03b`

The available models are:
- `seg03b`
- `seg06b`
- `seg1b`
- `depth03b`
- `depth06b`
- `depth1b`
- `depth2b`
- `normal03b`
- `normal06b`
- `normal1b`
- `normal2b`

## Usage

```python
import cv2
from imread_from_url import imread_from_url
from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType

# Load the model
config = SapiensConfig()
config.depth_type = SapiensDepthType.DEPTH_03B  # Disabled by default
config.normal_type = SapiensNormalType.NORMAL_1B  # Disabled by default
predictor = SapiensPredictor(config)

# Load the image
img = imread_from_url("https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/assets/test2.png?raw=true")

# Estimate the maps
result = predictor(img)

cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.imshow("Combined", result)
cv2.waitKey(0)
```

### SapiensPredictor
The `SapiensPredictor` class allows to run multiple tasks simultaneously. It has the following methods:
- `SapiensPredictor(config: SapiensConfig)` - Load the model with the specified configuration.
- `__call__(img: np.ndarray) -> np.ndarray` - Estimate the maps for the input image.

## Original Models
The original models are available at HuggingFace: https://huggingface.co/facebook/sapiens/tree/main/sapiens_lite_host
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/sapiens/blob/main/LICENSE)

## References
- **Sapiens**: https://github.com/facebookresearch/sapiens
- **Sapiens Lite**: https://github.com/facebookresearch/sapiens/tree/main/lite
- **HuggingFace Model**: https://huggingface.co/facebook/sapiens
