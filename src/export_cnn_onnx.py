import torch
import argparse
from models import DepthwiseCNN
import onnxruntime as ort
import numpy as np


def export_model(checkpoint_path: str, onnx_path: str):
    model = DepthwiseCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 15, 8, 8)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    ok = test_onnx_model(model, dummy_input, onnx_path)

    if ok:
        print(f"✅ Exported to {onnx_path}")
    else:
        print(f"❌ Exported invalid model")


def test_onnx_model(model, dummy_input, onnx_path):
    torch_model_output = model(dummy_input)

    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    ort_output = ort_outputs[0]

    return np.allclose(torch_model_output.detach().numpy(), ort_output, atol=1e-4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to export .onnx file")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output)