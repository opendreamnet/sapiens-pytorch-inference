import torch
import argparse
from sapiens_inference.common import download_hf_model, get_path_to_cache
from sapiens_inference import SapiensSegmentationType, SapiensNormalType, SapiensDepthType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type_dict = {
    "seg03b": SapiensSegmentationType.SEGMENTATION_03B,
    "seg06b": SapiensSegmentationType.SEGMENTATION_06B,
    "seg1b": SapiensSegmentationType.SEGMENTATION_1B,
    "normal03b": SapiensNormalType.NORMAL_03B,
    "normal06b": SapiensNormalType.NORMAL_06B,
    "normal1b": SapiensNormalType.NORMAL_1B,
    "normal2b": SapiensNormalType.NORMAL_2B,
    "depth03b": SapiensDepthType.DEPTH_03B,
    "depth06b": SapiensDepthType.DEPTH_06B,
    "depth1b": SapiensDepthType.DEPTH_1B,
    "depth2b": SapiensDepthType.DEPTH_2B
}

@torch.no_grad()
def export_model(model_name: str, filename: str):
    type = model_type_dict[model_name]
    path = download_hf_model(type.value)
    model = torch.jit.load(path)
    model = model.eval().to(device).to(torch.float32)
    input = torch.randn(1, 3, 1024, 768, dtype=torch.float32, device=device)  # Only this size seems to work well
    torch.onnx.export(model,
                      input,
                      get_path_to_cache(filename),
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=14,
                      input_names=["input"],
                      output_names=["output"])

def get_parser():
    parser = argparse.ArgumentParser(description="Export Sapiens models to ONNX")
    parser.add_argument("model", type=str, choices=model_type_dict.keys(), help="Model type to export")
    return parser

def run() -> None:
    args = get_parser().parse_args()
    export_model(args.model, f"{args.model}.onnx")
