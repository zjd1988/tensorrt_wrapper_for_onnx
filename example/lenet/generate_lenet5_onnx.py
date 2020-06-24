import torch
from torch import nn
from lenet5 import Lenet5
import os
import onnx
import onnxmltools
from onnxsim import simplify

def main():
    print('cuda device count: ', torch.cuda.device_count())
    torch.manual_seed(1234)
    net = Lenet5()
    net.eval()
    tmp = torch.ones(1, 1, 32, 32)
    execute_path = os.path.dirname(os.path.realpath(__file__))
    onnx_file = os.path.join(execute_path, "lenet.onnx")
    torch.onnx.export(net, tmp, onnx_file, export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    print('lenet out shape:', out.shape)
    print('lenet out:', out)

    model = onnx.load(onnx_file)
    model_simp, check = simplify(model)
    onnx_simplify_file = os.path.join(execute_path, "lenet_simplify.onnx")
    onnxmltools.utils.save_model(model_simp, onnx_simplify_file)

if __name__ == '__main__':
    main()

