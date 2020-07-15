import time
import execution
import cv2
import numpy as np

if __name__ == "__main__":
    # construct network
    test_input = np.ones((1, 1, 32, 32)).astype(np.float32)
    network = execution.Network()

    # 1 onnx model 
    onnx_file = "../example/lenet/lenet.onnx"
    onnx_node = execution.OnnxModelExecution(onnx_file, ["input"], ["output",])
    network.insert_node(onnx_node)

    # 7 inference network
    network.generate_topo_order(["input",])
    result = network.inference({"input": test_input}, ["output",])
    
    # 8 save network info to json file
    network.export_json()

