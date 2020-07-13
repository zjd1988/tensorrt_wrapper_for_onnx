import onnx
import onnxruntime
import json
import os
import sys
import numpy as np
import cv2
import copy

class Execution:
    execution_index = 0
    def __init__(self, execution_name, execution_input_name, execution_output_name):
        self.name = execution_name + "_" + str(Execution.execution_index)
        self.type = execution_name
        self.attr = {}
        self.input_name = []
        self.output_name = []
        Execution.execution_index += 1
        self.set_inputs(execution_input_name)
        self.set_outputs(execution_output_name)
    def get_inputs(self):
        return self.input_name
    def get_outputs(self):
        return self.output_name
    def set_inputs(self, names):
        self.input_name = names
    def set_outputs(self, names):
        self.output_name = names
    def get_execution_info(self):
        info = {}
        info["type"] = self.type
        info["attr"] = self.attr
        info["inputs"] = self.input_name
        info["outputs"] = self.output_name
        return info

class DataTypeConvertExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "DataTypeConvertExecution", execution_input_name, execution_output_name)

    def init_attr(self, attr):
        self.attr["convert_type"] = attr["convert_type"]

    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 1)
        if self.attr["convert_type"] == "ConvertUint8ToFloat32":
            result_data.extend(input_data[0].astype(np.float32))
        elif self.attr["convert_type"] == "ConvertUint8ToFloat16":
            result_data.extend(input_data[0].astype(np.float16))
        else:
            print("not supported type {}".format(attr["convert_type"]))
            assert False
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class DataFormatConvertExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "DataFormatConvertExecution", execution_input_name, execution_output_name)

    def init_attr(self, attr):
        self.attr["convert_type"] = attr["convert_type"]

    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 1)
        if self.attr["convert_type"] == "RGB2BGR":
            result_data.append(cv2.cvtColor(input_data[0], cv2.COLOR_RGB2BGR))
        elif self.attr["convert_type"] == "BGR2RGB":
            result_data.append(cv2.cvtColor(input_data[0], cv2.COLOR_BGR2RGB))
        elif self.attr["convert_type"] == "RGB2GRAY":
            result_data.append(cv2.cvtColor(input_data[0], cv2.COLOR_RGB2GRAY))
        elif self.attr["convert_type"] == "BGR2GRAY":
            result_data.append(cv2.cvtColor(input_data[0], cv2.COLOR_BGR2GRAY))
        else:
            print("not supported type {}".format(attr["convert_type"]))
            assert False
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class ReshapeExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "ReshapeExecution", execution_input_name, execution_output_name)

    def init_attr(self, attr):
        self.attr["shape"] = attr["shape"]

    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 1)
        shape = self.attr["shape"]
        result_data.append(np.reshape(input_data[0], shape))
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class TransposeExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "TransposeExecution", execution_input_name, execution_output_name)


    def init_attr(self, attr):
        self.attr["perm"] = attr["perm"]

    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 1)
        if "perm" in self.attr.keys():
            perm = self.attr["perm"]
            result_data.append(input_data[0].transpose(perm))
        else:
            print("have no perm attr!")
            assert False
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class NormalizationExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "NormalizationExecution", execution_input_name, execution_output_name)
        self.attr["alpha"] = 0.0
        self.attr["beta"] = 1.0


    def init_attr(self, attr):
        for attr_key, attr_value in attr.items():
            if attr_key in self.attr:
                self.attr[attr_key] = attr_value
            else:
                pass

    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 1)
        alpha = self.attr["alpha"]
        beta = self.attr["beta"]
        normalize_result = (input_data[0] - alpha) / beta
        result_data.append(normalize_result.astype(np.float32))
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class OnnxExecution(Execution):
    def __init__(self, onnx_file, execution_input_name, execution_output_name):
        Execution.__init__(self, "OnnxExecution", execution_input_name, execution_output_name)
        self.session = onnxruntime.InferenceSession(onnx_file)
        self.attr["onnx_file"] = onnx_file

    def init_attr(self, attr):
        pass

    def execute(self, input_data):
        input_name = self.get_inputs()
        assert(len(input_data) == len(input_name))
        output_name = self.get_outputs()
        results = self.session.run(output_name, dict(zip(input_name,input_data)))
        return dict(zip(output_name,results))


class YoloNMSExecution(Execution):
    def __init__(self, execution_input_name, execution_output_name):
        Execution.__init__(self, "YoloNMSExecution", execution_input_name, execution_output_name)
        self.attr["img_height"]  = 512
        self.attr["img_width"]   = 384
        self.attr["conf_thresh"] = 0.3
        self.attr["iou_thresh"]  = 0.6
        self.attr["topk"]        = 200


    def init_attr(self, attr):
        for attr_key, attr_value in attr.items():
            if attr_key in self.attr:
                self.attr[attr_key] = attr_value
            else:
                pass
    
    def xywh2xyxy(self, x, strideX, strideY):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, 0] = x[:, 0] - x[:, 2] * strideX / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] * strideY / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] * strideX / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] * strideY / 2  # bottom right y
        return y

    def non_max_suppression(self, input_data):
        inputs = self.get_inputs
        strideX = self.attr["img_width"] / 32
        strideY = self.attr["img_height"] / 32
        conf_thresh = self.attr["conf_thresh"]
        iou_thresh = self.attr["iou_thresh"]
        classes = input_data[0]
        boxes = input_data[1]
        boxes_size = boxes.shape[0]
        class_num = classes.shape[1]
        boxes = self.xywh2xyxy(boxes, strideX, strideY)
        class_result = np.nonzero(classes > conf_thresh)
        box_match_thresh_index = class_result[0]
        class_index = class_result[1]
        scores = classes[np.nonzero(classes > conf_thresh)]

        valid_boxes = boxes[box_match_thresh_index,:]
        keep = []
        if len(box_match_thresh_index) > 0:
            x1 = valid_boxes[:, 0]
            y1 = valid_boxes[:, 1]
            x2 = valid_boxes[:, 2]
            y2 = valid_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            while order.size > 0:
                i = order[0] # pick maxmum iou box
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
                h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= iou_thresh)[0]
                order = order[inds + 1]
        
        results = []
        if keep != []:
            results.append(valid_boxes[keep,:])
            results.append(class_index[keep])
        else:
            results = [[], []]
        return results

        
    def execute(self, input_data):
        result_data = []
        assert(len(input_data) == 2)
        result_data.extend(self.non_max_suppression(input_data))
        output_name = self.get_outputs()
        return dict(zip(output_name,result_data))

class Network:
    net_index = 0
    def __init__(self, net_name = ""):
        if net_name == "":
            self.net_name = "Network_" + str(Network.net_index)
            Network.net_index += 1
            self.nodes = {}
            self.nodes_names = set()
            self.topo_order = []
            self.blobs = {}

    def insert_node(self, node):
        self.nodes[node.name] = node
        self.nodes_names.add(node.name)
    
    def get_blob_data(self, input_names):
        result = []
        for i in range(len(input_names)):
            result.append(self.blobs[input_names[i]])
        return result
    
    def set_blob_data(self, inputs_data):
        for name, data in inputs_data.items():
            self.blobs[name] = data

    def generate_topo_order(self, input_names):
        self.topo_order = []
        temp_names = input_names
        all_nodes = copy.deepcopy(self.nodes_names)
        while len(all_nodes) != 0:
            topo_flag = False
            clone_nodes = copy.deepcopy(all_nodes)
            for node_name in all_nodes:
                node_info = self.nodes[node_name]
                node_input_names = node_info.get_inputs()
                exist_flag = True
                for i in range(len(node_input_names)):
                    if node_input_names[i] not in temp_names:
                        exist_flag = False
                        break
                if exist_flag == True:
                    self.topo_order.append(node_name)
                    temp_names.extend(node_info.get_outputs())
                    topo_flag = True
                    clone_nodes.remove(node_name)
            all_nodes = copy.deepcopy(clone_nodes)
            if topo_flag == False:
                print("topo order generate fail!\n")
                assert topo_flag

    def inference(self, inputs_data, outputs_data):
        self.set_blob_data(inputs_data)
        for index in range(len(self.topo_order)):
            curr_node = self.nodes[self.topo_order[index]]
            blob_names = curr_node.get_inputs()
            curr_inputs = self.get_blob_data(blob_names)
            temp_results = curr_node.execute(curr_inputs)
            self.set_blob_data(temp_results)
        
        result = self.get_blob_data(outputs_data)
        return result

    def export_json(self, file_name=""):
        json_result = {}
        execution_node_info = {}
        for i in range(len(self.topo_order)):
            node = self.nodes[self.topo_order[i]]
            execution_node_info[self.topo_order[i]] = node.get_execution_info()

        if file_name == "":
            file_name = "net_inference.json"
        json_result["node_info"] = execution_node_info
        json_result["topo_order"] = self.topo_order
        json_str = json.dumps(json_result)
        with open(file_name, 'w') as f:
            f.write(json_str)
