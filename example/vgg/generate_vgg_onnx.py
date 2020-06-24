import torch
import torchvision
import os
import struct
import onnx
import onnxmltools
from onnxsim import simplify

def main():
    torch.manual_seed(1234)
    net = torchvision.models.vgg11(pretrained=True)
    net = net.eval()
    print(net)
    tmp = torch.ones(1, 3, 224, 224)
    execute_path = os.path.dirname(os.path.realpath(__file__))
    onnx_file = os.path.join(execute_path, "vgg.onnx")
    torch.onnx.export(net, tmp, onnx_file, export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    print('vgg out shape:', out.shape)
    print('vgg out:', out)
    

    model = onnx.load(onnx_file)
    model_simp, check = simplify(model)
    onnx_simplify_file = os.path.join(execute_path, "vgg_simplify.onnx")
    onnxmltools.utils.save_model(model_simp, onnx_simplify_file)

if __name__ == '__main__':
    main()

