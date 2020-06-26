# prepare
* pytorch 1.4+

* git clone https://github.com/ultralytics/yolov3.git

* download yolov3.pt from https://drive.google.com/drive/folders/1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0 , and place to yolov3/weights dir

# convert model to onnx
1. modify ONNX_EXPORT=True int models.py

2. modify torch.onnx.export(xxxxx, opset_version=10, xxxxx) in detect.py

3. python3 detect.py --cfg cfg/yolov3-spp.cfg --weights weights/yolov3-spp.pt, you will get a yolov3-spp.onnx

4. python3 simplify_yolov3_spp_onnx.py to get yolov3-spp_simplify.onnx

5. cd yolov3-spp, and run sh onnx_to_tensorrt.sh, generate net_graph.json net_weights.bin for c++ code.

