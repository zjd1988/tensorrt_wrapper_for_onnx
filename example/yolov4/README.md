# prepare
* pytorch 1.4+

* git clone https://github.com/ultralytics/yolov3.git

* download yolov4.pt from https://drive.google.com/drive/folders/1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0 , and place to yolov3/weights dir

# convert model to onnx
1. replace xxx/example/yolov4/models.py in https://github.com/ultralytics/yolov3.git

2. python3 detect.py --cfg cfg/yolov4.cfg --weights weights/yolov4.pt, you will get a yolov4.onnx

3. python3 simplify_yolov4_onnx.py to get yolov4_simplify.onnx

4. cd xxx/example/yolov4/, and run sh onnx_to_tensorrt.sh, generate net_graph.json net_weights.bin for c++ code.

