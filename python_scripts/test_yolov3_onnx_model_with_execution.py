import time
import execution
import cv2


if __name__ == "__main__":
    test_img_file = "../example/yolov3/bus.jpg"
    test_input = cv2.imread(test_img_file)
    # construct network
    network = execution.Network()
    # 1 brg to rgb
    bgr2rgb_node = execution.DataFormatConvertExecution(["bgr_image",], ["rgb_image",])
    attr = {}
    attr["convert_type"] = "BGR2RGB"
    bgr2rgb_node.init_attr(attr)
    network.insert_node(bgr2rgb_node)

    # 2 reshape rgb to 1 * h * w * c
    inputs = bgr2rgb_node.get_outputs()
    reshape_node = execution.ReshapeExecution(inputs, ["reshape_image",])
    attr = {}
    shape = [1, ]
    img_shape = test_input.shape
    shape.extend(list(img_shape))
    attr["shape"] = shape
    reshape_node.init_attr(attr)
    network.insert_node(reshape_node)

    # 3 transpose nhwc to nchw
    inputs = reshape_node.get_outputs()
    transpose_node = execution.TransposeExecution(inputs, ["transpose_img",])
    attr = {}
    attr["perm"] = [0, 3, 1, 2]
    transpose_node.init_attr(attr)
    network.insert_node(transpose_node)

    # 4 normalization 
    inputs = transpose_node.get_outputs()
    normalization_node = execution.NormalizationExecution(inputs, ["images",])
    attr = {}
    attr["alpha"] = 0.0
    attr["beta"] = 255.0
    attr["bias"] = 0.0
    normalization_node.init_attr(attr)
    network.insert_node(normalization_node)

    # 5 onnx model 
    inputs = normalization_node.get_outputs()
    onnx_file = "../example/yolov3/yolov3-tiny.onnx"
    onnx_node = execution.OnnxModelExecution(onnx_file, inputs, ["classes", "boxes"])
    network.insert_node(onnx_node)

    # 6 yolo nms
    inputs = onnx_node.get_outputs()
    nms_node = execution.YoloNMSExecution(inputs, ["nms_number", "nms_boxes", "nms_classes"])
    network.insert_node(nms_node)

    # 7 inference network
    outputs = nms_node.get_outputs()
    network.generate_topo_order(["bgr_image",])
    result = network.inference({"bgr_image": test_input}, outputs)
    
    # 8 save network info to json file
    network.export_json()

