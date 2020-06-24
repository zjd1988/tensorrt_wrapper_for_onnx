# tensorrt_wrapper_for_onnx
coding ...

# requiemets
1 cuda 10.x

2 tensorrt 7.x

3 python 3.5+

4 pytorch 1.5(cpu/gpu version)


# verified models
1 lenet        ./example/lenet/lenet_simplify.onnx              ------- verified(fp32)

2 mobilenet_v2 ./example/mobilenet_v2_simplify.onnx             ------- verified(fp32)

3 vgg          ./example/vgg/vgg_simplify.onnx                  ------- verified(fp32)

# limitations
1 only support batch size 1

2 only support fp32/fp16 model

3 onnx model supports op version 11+



# reference
1 https://github.com/saurabh-shandilya/onnx-utils.git

2 https://github.com/wang-xinyu/tensorrtx.git

3 https://github.com/onnx/onnx-tensorrt.git

4 https://github.com/microsoft/onnxruntime.git
