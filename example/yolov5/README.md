# prepare
* pytorch 1.4+

* git clone https://github.com/ultralytics/yolov5.git

* download yolov5s.pt/yolov5m.pt/yolov5l.pt/yolov5x.pt from https://drive.google.com/drive/folders/1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J , and place to yolov5/weights dir

# convert model to onnx
1. export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1

2. cp yolov5s.onnx to tensorrt_wrapper_for_onnx/example/yolov5 dir

3. cd example/yolov5 ,and run python3 simplify_yolov5s_onnx.py to get yolov5s_simplify.onnx

4. run sh onnx_to_tensorrt.sh, generate net_graph.json net_weights.bin for c++ code.

