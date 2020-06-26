import torch
import torchvision
import os
import onnx
import onnxmltools
from onnxsim import simplify
from torchsummary import summary

def main():
    torch.manual_seed(1234)
    net = torchvision.models.squeezenet1_1(pretrained=True)
    net = net.eval()
    print(net)
    tmp = torch.ones(1, 3, 227, 227)
    execute_path = os.path.dirname(os.path.realpath(__file__))
    onnx_file = os.path.join(execute_path, "squeezenet.onnx")
    torch.onnx.export(net, tmp, onnx_file, export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    summary(net, (3, 227, 227))
    print('squeezenet out shape:', out.shape)
    print('squeezenet out:', out)
    

    model = onnx.load(onnx_file)
    model_simp, check = simplify(model)
    onnx_simplify_file = os.path.join(execute_path, "squeezenet_simplify.onnx")
    onnxmltools.utils.save_model(model_simp, onnx_simplify_file)

if __name__ == '__main__':
    main()

