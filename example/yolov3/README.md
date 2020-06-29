# prepare
* pytorch 1.4+

* git clone https://github.com/ultralytics/yolov3.git

* download yolov3.pt from https://drive.google.com/drive/folders/1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0 , and place to yolov3/weights dir

# convert model to onnx
1. cd yolov3(https://github.com/ultralytics/yolov3.git)

2. modify ONNX_EXPORT=True int models.py

3. modify torch.onnx.export(xxxxx, opset_version=10, xxxxx) in detect.py

4. python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.pt, you will get a yolov3.onnx

5. cd xxx/example/yolov3/

6. python3 simplify_yolov3_onnx.py to get yolov3_simplify.onnx

7. and run sh onnx_to_tensorrt.sh, generate net_graph.json net_weights.bin for c++ code.

