import torch
from torch import nn
from lenet5 import Lenet5
import os
import struct

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('lenet5.pth')
    net = net.to('cuda:0')
    net.eval()
    
    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    torch.onnx.export(net, tmp, "lenet5.onnx", export_params=True, opset_version=10, input_names = ['input'], output_names = ['output'])
    out = net(tmp)
    print('lenet out:', out)

if __name__ == '__main__':
    main()

