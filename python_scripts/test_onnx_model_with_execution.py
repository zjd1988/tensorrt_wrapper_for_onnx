import time
import execution
import cv2

if __name__ == "__main__":
    test_img_file = "../example/yolov3/bus.jpg"
    test_input = cv2.imread(test_img_file)
    # construct session
    session_info = execution.construct_onnx_session("../example/yolov3/yolov3-tiny.onnx", pre_execution, post_execution)
    input_data = {}
    input_data[session_info.inputs_name[0]]  = test_input
    # prepare execution
    pre_execution = []
    post_execution = []
    temp_execution = execution.create_datatype_convert_execution("ConvertUint8ToFloat32")
    pre_execution.append(temp_execution)
    temp_execution = execution.create_normalization_execution()
    pre_execution.append(temp_execution)
    temp_execution = execution.create_yolo_nms_execution(512, 384)
    post_execution.append(temp_execution)
    # run session with results(execution result and net inference result)
    session_results = execution.run_session(session_info, input_data)
    
    print(session_results[0])

