# tensorrt_wrapper_for_onnx
coding ...

# requiemets
* cuda 10.x

* tensorrt 7.x

* python 3.5+

* pytorch 1.5(cpu/gpu version)


# verified models
* nvidia-gpu  GTX1060(3GB)

1. lenet  ------------- ./example/lenet/lenet_simplify.onnx ---------------------- verified(fp16/fp32)

2. mobilenet_v2 ------- ./example/mobilenet_v2_simplify.onnx --------------------- verified(fp32)

3. vgg ---------------- ./example/vgg/vgg_simplify.onnx -------------------------- verified(fp32)

4. squeezenet --------- ./example/squeezenet/squeeze_simplify.onnx --------------- verified(fp32)

5. yolov3 ------------- ./example/yolov3/yolov3_simplify.onnx --------------------- verified(fp32)

6. yolov3-tiny -------- ./example/yolov3/yolov3-tiny_simplify.onnx ---------------- verified(fp32)

7. yolov3-spp --------- ./example/yolov3/yolov3-spp_simplify.onnx ----------------- verified(fp32)

# step 1
1. cd tesensorrt_wrapper_for_onnx

2. (optional) download jsoncpp-00.11.0 from https://github.com/open-source-parsers/jsoncpp/releases  

3. compile jsoncpp-00.11.0 follow https://github.com/open-source-parsers/jsoncpp/wiki/Building

4. mkdir build && cd build

5. cmake .. && make -j4

# step 2
1. cd ./example/lenet

2. run generate_lenet5_onnx.py, generate lenet_simplify.onnx

3. run sh onnx_to_tensorrt.sh , generate net_graph.json(contains network's node info) / net_weights.bin(contains network weights)

4. run ./build/lenet_example 0 , generate net.engine file(used for tensorrt inference)

5. run ./build/lenet_example 1 , run tensorrt inference wiht net.engine

# how to add new ops



# limitations
1. only support batch size 1

2. only support fp32/fp16 model

3. onnx model supports op version 11+


# reference
1. https://github.com/saurabh-shandilya/onnx-utils.git

2. https://github.com/wang-xinyu/tensorrtx.git

3. https://github.com/onnx/onnx-tensorrt.git

4. https://github.com/microsoft/onnxruntime.git
