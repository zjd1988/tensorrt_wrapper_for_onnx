import torch
import torchvision
import os
import onnx
import onnxmltools
from onnxsim import simplify
from torchsummary import summary

def main():
    torch.manual_seed(1234)
    tmp = torch.ones(1, 3, 512, 384)
    execute_path = os.path.dirname(os.path.realpath(__file__))
    onnx_file = os.path.join(execute_path, "yolov3-tiny.onnx")

    model = onnx.load(onnx_file)
    model_simp, check = simplify(model)
    onnx_simplify_file = os.path.join(execute_path, "yolov3-tiny_simplify.onnx")
    onnxmltools.utils.save_model(model_simp, onnx_simplify_file)

if __name__ == '__main__':
    main()

