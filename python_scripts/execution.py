import onnx
import onnxruntime
import json
import os
import sys
import numpy as np

onnx_data_type = {}
onnx_data_type["UNDEFINED"] = 0
onnx_data_type["FLOAT"] = 1
onnx_data_type["UINT8"] = 2
onnx_data_type["INT8"] = 3
onnx_data_type["UINT16"] = 4
onnx_data_type["INT16"] = 5
onnx_data_type["INT32"] = 6
onnx_data_type["INT64"] = 7
onnx_data_type["STRING"] = 8
onnx_data_type["BOOL"] = 9
onnx_data_type["FLOAT16"] = 10
onnx_data_type["DOUBLE"] = 11
onnx_data_type["UINT32"] = 12
onnx_data_type["UINT64"] = 13
onnx_data_type["COMPLEX64"] = 14
onnx_data_type["COMPLEX128"] = 15
onnx_data_type["BFLOAT16"] = 16

attr_type = {}
attr_type["UNDEFINED"] = 0
attr_type["FLOAT"] = 1
attr_type["INT"] = 2
attr_type["STRING"] = 3
attr_type["TENSOR"] = 4
attr_type["GRAPH"] = 5
attr_type["FLOATS"] = 6
attr_type["INTS"] = 7
attr_type["STRINGS"] = 8
attr_type["TENSORS"] = 9
attr_type["GRAPHS"] = 10
attr_type["SPARSE_TENSOR"] = 11
attr_type["SPARSE_TENSORS"] = 12

PRE_EXECUTION_RUN_FUNC = {}
POST_EXECUTION_RUN_FUNC = {}
EXECUTION_RUN_FUNC = {}
# execution_type
# attribute
# supported type as follows: 
datatype_convert_set = set()
datatype_convert_set.add("ConvertUint8ToFloat32")
datatype_convert_set.add("ConvertUint8ToFloat16")
def datatype_convert_func(input_data, attr):

def create_datatype_convert_execution(convert_type):
    execution = {}
    attr = {}
    if convert_type in datatype_convert_set:
        attr["convert_type"] = convert_type
    else:
        print("not supported type {}".format(convert_type))
        assert False
    execution["type"] = "DataTypeConvert"
    execution["attr"] = attr
    execution["func"] = datatype_convert_func


# supported type as follows:
format_convert_set = set()
format_convert_set.add("RGB2BGR")
format_convert_set.add("BGR2RGB")
format_convert_set.add("RGB2GRAY")
format_convert_set.add("BGR2GRAY")
def create_format_convert_execution(convert_type):
    execution = {}
    attr = {}
    if convert_type in format_convert_set:
        attr["convert_type"] = convert_type
    else:
        print("not supported type {}".format(convert_type))
        assert False
    execution["type"] = "FormatConvert"
    execution["attr"] = attr
    return execution


def create_transpose_execution(axes):
    execution = {}
    attr = {}
    axes_len = len(axes)
    transpose_axes = []
    for i in range(axes_len):
        transpose_axes.append(int(axes[0]))
    attr["axes"] = transpose_axes
    execution["type"] = "Transpose"
    execution["attr"] = attr
    return execution


def create_yolo_nms_execution(img_height, img_width, conf_thresh = 0.3, iou_thresh = 0.6, topk = 200):
    execution = {}
    attr = {}
    attr["img_height"]  = int(img_height)
    attr["img_width"]   = int(img_width)
    attr["conf_thresh"] = float(conf_thresh)
    attr["iou_thresh"]  = float(iou_thresh)
    attr["topk"]        = int(topk)
    execution["attr"] = attr
    return execution


def create_normalization_execution(alpha = 0.0, beta = 255.0):
    execution = {}
    attr = {}
    attr["alpha"] = float(alpha)
    attr["beta"] = float(beta)
    execution["type"] = "Normalization"
    execution["attr"] = attr
    return execution


def construct_onnx_session(onnx_file, pre_execution = [], post_execution = []):
    onnx_session = {}
    session = onnxruntime.InferenceSession(onnx_file)
    inputs = session.get_inputs()
    inputs_name = []
    for i in len(inputs):
        input_name = session.get_inputs()[0].name
        inputs_name.append(input_name)
    
    if len(pre_execution) == 0:
        temp_execution = create_datatype_convert_execution("ConvertUint8ToFloat32")
        pre_execution.append(temp_execution)

    onnx_session["session"] = session
    onnx_session["inputs_name"] = inputs_name
    onnx_session["pre_execution"] = pre_execution
    onnx_session["post_execution"] = post_execution

    return onnx_session


def run_session(session_info, input_data):
    pre_len = len(session_info["pre_execution"])
    pre_results = []
    post_results = []
    net_inference_results = []
    for i in range(pre_len):
        pre_execution = session_info["pre_execution"][i]
        result = pre_execution["func"](input_data)
        pre_results.append(result)
        input_data = result

    net_inference_results = onnx_session["session"].run([], input_data)
    input_data = net_inference_results

    for i in range(pre_len):
        post_execution = session_info["post_execution"][i]
        result = post_execution["func"](input_data)
        post_results.append(result)
        input_data = result

    results = {}
    results["pre_results"] = pre_results
    results["net_results"] = net_inference_results
    results["post_results"] = post_results