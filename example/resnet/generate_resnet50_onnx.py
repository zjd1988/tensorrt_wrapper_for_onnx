import torch
import torchvision
import os
import onnx
import onnxmltools
from onnxsim import simplify

def main():
    net = torchvision.models.resnet50(pretrained=True)
    net = net.eval()
    print(net)
    tmp = torch.ones(1, 3, 224, 224)
    execute_path = os.path.dirname(os.path.realpath(__file__))
    onnx_file = os.path.join(execute_path, "resnet50.onnx")
    torch.onnx.export(net, tmp, onnx_file, export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    print('resnet50 out shape:', out.shape)
    print('resnet50 out:', out)
    
    model = onnx.load(onnx_file)
    model_simp, check = simplify(model, skip_fuse_bn = True)
    # model_simp, check = simplify(model) # get errors https://github.com/daquexian/onnx-simplifier/issues/53
    onnx_simplify_file = os.path.join(execute_path, "resnet50_simplify.onnx")
    onnxmltools.utils.save_model(model_simp, onnx_simplify_file)
   

if __name__ == '__main__':
    main()

