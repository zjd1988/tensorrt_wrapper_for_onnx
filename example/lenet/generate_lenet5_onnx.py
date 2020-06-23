import torch
from torch import nn
from lenet5 import Lenet5

def main():
    print('cuda device count: ', torch.cuda.device_count())
    torch.manual_seed(1234)
    net = Lenet5()
    # net = net.to('cuda:0')
    net.eval()
    # tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    tmp = torch.ones(1, 1, 32, 32)
    torch.onnx.export(net, tmp, "lenet.onnx", export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    print('lenet out shape:', out.shape)
    print('lenet out:', out)
    # torch.save(net, "lenet.pth")

if __name__ == '__main__':
    main()

