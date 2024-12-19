# note
will update to cuda-12.x and tensorrt-10.x

# tensorrt_wrapper_for_onnx

* easily modifying and converting onnx models to tensorrt engine files

* easily adding new ops

* easily infering the engine files

* easily integrating with your own projects

# requiemets
* cuda 10.x

* tensorrt 7.x

* python 3.5+

* pytorch 1.4+(cpu/gpu version)


# verified models
|      model      |      int8      |      fp16      |      fp32      |     device     |
 -----------------|----------------|----------------|----------------|----------------
|      lenet      |       no       |       yes      |      yes       |                |
|   mobilenet_v2  |       no       |       yes      |      yes       |                |
|       vgg       |       no       |       yes      |      yes       |                |
|    squeezenet   |       no       |       yes      |      yes       |                |
|      yolov3     |       no       |       yes      |      yes       |                |
|   yolov3-tiny   |       no       |       yes      |      yes       |                |
|    yolov3-spp   |       no       |       yes      |      yes       |                |
|      yolov4     |       no       |       yes      |      yes       |                |
|      yolov5     |       no       |       yes      |      yes       |                |

# step 1
1. cd tesensorrt_wrapper_for_onnx

2. (optional) download jsoncpp-1.9.6: Bugfixes from https://github.com/open-source-parsers/jsoncpp/releases  

3. compile jsoncpp-1.9.6 follow https://github.com/open-source-parsers/jsoncpp/wiki/Building

4. mkdir build && cd build

5. cmake .. && make -j4

# step 2
1. cd ./example/lenet

2. run generate_lenet5_onnx.py, generate lenet_simplify.onnx

3. run sh onnx_to_tensorrt.sh , generate net_graph.json(contains network's node info) / net_weights.bin(contains network weights)

4. run ./build/lenet_example (modify #define SAVE_ENGINE 1, in lenet_example.cpp) , generate net.engine file(used for tensorrt inference)

5. run ./build/lenet_example (modify #define SAVE_ENGINE 0, in lenet_example.cpp), run tensorrt inference wiht net.engine

6. if you want to use fp16, then modify #define FP16_FLAG true, in lenet_example.cpp

# how to add new ops



# todo lists
* add GPU allocator to manage device memory

* add common plugin implement(NMS mish ...)

* add new models(alexnet shufflenet ...)

* add custom network pre/post process interfaces

# limitations
1. only support batch size 1

2. only support fp32/fp16 model

3. onnx model supports op version 10+


# reference
1. https://github.com/saurabh-shandilya/onnx-utils.git

2. https://github.com/wang-xinyu/tensorrtx.git

3. https://github.com/onnx/onnx-tensorrt.git

4. https://github.com/microsoft/onnxruntime.git
