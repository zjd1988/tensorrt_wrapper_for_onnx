import time
import execution
import numpy as np
import cv2


if __name__ == "__main__":
    test_img_file = "../example/hfnet/gray_test.bmp"
    color_img = cv2.imread(test_img_file)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).reshape((1, 720,1280,1))
    test_input = np.zeros((1,721,1281,1)).astype(np.uint8)
    test_input[:,0:720,0:1280,:] = gray_img

    # construct network
    network = execution.Network()
    # 1 uint8 to float32
    uint8ToFloat32_node = execution.DataTypeConvertExecution(["gray_image",], ["prefix/image:0",])
    attr = {}
    attr["convert_type"] = "ConvertUint8ToFloat32"
    uint8ToFloat32_node.init_attr(attr)
    network.insert_node(uint8ToFloat32_node)

    # 2 onnx model 
    inputs = uint8ToFloat32_node.get_outputs()
    onnx_file = "../example/hfnet/hfnet_edit.onnx"
    onnx_node = execution.OnnxModelExecution(onnx_file, inputs, ["prefix/pred/global_head/l2_normalize:0",
         "prefix/pred/local_head/descriptor/Mul_1:0", "prefix/pred/Reshape:0", "prefix/pred/keypoint_extraction/Greater_new:0"])
    network.insert_node(onnx_node)

    # 3 resample 
    inputs = onnx_node.get_outputs()
    resample_node = execution.HFnetResampleExecution(inputs, ["global_desc", "local_desc",])
    network.insert_node(resample_node)

    # 4 inference network
    outputs = resample_node.get_outputs()
    network.generate_topo_order(["gray_image",])
    result = network.inference({"gray_image": test_input}, outputs)
    
    # 5 save network info to json file
    network.export_json()

