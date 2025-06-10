import torch
import argparse
from models import HalfKAModel, HalfKAInputProcessor, HalfKABucketEvaluator
from data_sets.halfka_dataset import HalfKADataset
import onnxruntime as ort
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to export .onnx file")
    return parser.parse_args()


def export_model(checkpoint_path: str, onnx_path: str):
    model = HalfKAModel(checkpoint_path=checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    input_processor = HalfKAInputProcessor(model)
    bucket_evaluators = [HalfKABucketEvaluator(model, i) for i in range(8)]


    processor_dummy_input_own = torch.randn(1, 520)
    processor_dummy_input_opp = torch.randn(1, 520)

    dummy_x_1024, dummy_avg_score = input_processor(processor_dummy_input_own, processor_dummy_input_opp)

    onnx_path_input_processor = onnx_path.replace(".onnx", "_input_processor.onnx")
    onnx_path_bucket_evaluators = [onnx_path.replace(".onnx", f"_bucket_evaluator_{i}.onnx") for i in range(8)]

    embeddings_own = model.precomputed_own.detach().clone().cpu()
    embeddings_opp = model.precomputed_opp.detach().clone().cpu()


    np.save(onnx_path.replace(".onnx", "_embeddings_own.npy"), embeddings_own.numpy())
    np.save(onnx_path.replace(".onnx", "_embeddings_opp.npy"), embeddings_opp.numpy())

    print(f"✅ Exported embeddings to {onnx_path.replace('.onnx', '_embeddings_own.npy')} and {onnx_path.replace('.onnx', '_embeddings_opp.npy')}")

    torch.onnx.export(
        input_processor,
        (processor_dummy_input_own, processor_dummy_input_opp),
        onnx_path_input_processor,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["own_sum", "opp_sum"],
        output_names=["x_1024", "avg_score"],
        dynamic_axes={
            "own_sum": {0: "batch_size"},
            "opp_sum": {0: "batch_size"},
            "x_1024": {0: "batch_size"},
            "avg_score": {0: "batch_size"},
        }
    )

    for i, bucket_evaluator in enumerate(bucket_evaluators):
        torch.onnx.export(
            bucket_evaluator,
            (dummy_x_1024, dummy_avg_score),
            onnx_path_bucket_evaluators[i],
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["x_1024", "avg_score"],
            output_names=["output"],
            dynamic_axes={
                "x_1024": {0: "batch_size"},
                "avg_score": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        )

    ok_input_processor = test_onnx_processor(input_processor, (processor_dummy_input_own, processor_dummy_input_opp), onnx_path_input_processor)
    if not ok_input_processor:
        print(f"❌ Exported invalid input processor")
        return

    print(f"✅ Exported input processor to {onnx_path_input_processor}")

    for i, bucket_evaluator in enumerate(bucket_evaluators):
        ok_bucket_evaluator = test_onnx_bucket(bucket_evaluator, (dummy_x_1024, dummy_avg_score), onnx_path_bucket_evaluators[i])
        if not ok_bucket_evaluator:
            print(f"❌ Exported invalid bucket evaluator {i}")
            return

    print(f"✅ Exported bucket evaluators to {onnx_path_bucket_evaluators}")


def test_onnx_processor(model, dummy_input, onnx_path):
    torch_model_output = model(dummy_input[0], dummy_input[1])

    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input[0].numpy(), ort_session.get_inputs()[1].name: dummy_input[1].numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    ort_output_x_1024 = ort_outputs[0]
    ort_output_avg_score = ort_outputs[1]

    return np.allclose(torch_model_output[0].detach().numpy(), ort_output_x_1024, atol=1e-4) and \
    np.allclose(torch_model_output[1].detach().numpy(), ort_output_avg_score, atol=1e-4)


def test_onnx_bucket(model, dummy_input, onnx_path):
    torch_model_output = model(dummy_input[0], dummy_input[1])

    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input[0].numpy(), ort_session.get_inputs()[1].name: dummy_input[1].detach().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    ort_output = ort_outputs[0]

    return np.allclose(torch_model_output.detach().numpy(), ort_output, atol=1e-4)


if __name__ == "__main__":
    args = parse_args()
    export_model(args.checkpoint, args.output)