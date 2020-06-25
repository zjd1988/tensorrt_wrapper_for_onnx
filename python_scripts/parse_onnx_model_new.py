import onnx
import json
import os
import sys
import numpy as np

node_func = {}
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

def get_node_attribute(node_attr, attributes):
    for i in range(len(node_attr)):
        if node_attr[i].type == attr_type["INT"]:
            attributes[node_attr[i].name] = [node_attr[i].i]
        elif node_attr[i].type == attr_type["INTS"]:
            attributes[node_attr[i].name] = list(node_attr[i].ints)
        elif node_attr[i].type == attr_type["FLOAT"]:
            attributes[node_attr[i].name] = [node_attr[i].f]
        elif node_attr[i].type == attr_type["FLOATS"]:
            attributes[node_attr[i].name] = list(node_attr[i].floats)            
        else:
            pass

def get_node_raw_data(node_ele):
    count = 0
    raw_data = []
    if len(node_ele.raw_data) == 0:
        if node_ele.data_type == onnx_data_type["FLOAT"]:
            raw_data = np.array(node_ele.float_data).astype(np.float32)
            raw_data = raw_data.tobytes()
            count = len(raw_data)
        elif node_ele.data_type == onnx_data_type["FLOAT16"]:
            raw_data = np.array(node_ele.float_data).astype(np.float16)
            raw_data = raw_data.tobytes()
            count = len(raw_data)
    else:
        count = len(node_ele.raw_data)
        raw_data = node_ele.raw_data
    return count, raw_data


NODE_PARSE_FUNC = {}
def node_parse_without_attr(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    return node
NODE_PARSE_FUNC["node_without_attr"] = node_parse_without_attr

def node_parse_with_attr(node_info, weights_info):
    node = {}
    inputs = list(node_info.input)
    outputs = list(node_info.output)
    op_type = node_info.op_type
    attributes = {}
    node_attr = node_info.attribute
    get_node_attribute(node_attr, attributes)

    node["inputs"] = inputs
    node["outputs"] = outputs
    node["op_type"] = op_type
    node["attributes"] = attributes
    return node    
NODE_PARSE_FUNC["node_with_attr"] = node_parse_with_attr

NODE_WITHOUT_ATTRIBUTE = set()
NODE_WITH_ATTRIBUTE = set()
def init_node_without_attr():
    NODE_WITHOUT_ATTRIBUTE.add("Clip")
    NODE_WITHOUT_ATTRIBUTE.add("Add")
    NODE_WITHOUT_ATTRIBUTE.add("Sub")
    NODE_WITHOUT_ATTRIBUTE.add("Mul")
    NODE_WITHOUT_ATTRIBUTE.add("Div")
    NODE_WITHOUT_ATTRIBUTE.add("Exp")
    NODE_WITHOUT_ATTRIBUTE.add("Abs")
    NODE_WITHOUT_ATTRIBUTE.add("Sqrt")
    NODE_WITHOUT_ATTRIBUTE.add("Reciprocal")
    NODE_WITHOUT_ATTRIBUTE.add("Reshape")
    NODE_WITHOUT_ATTRIBUTE.add("Max")
    NODE_WITHOUT_ATTRIBUTE.add("GlobalAveragePool")
    NODE_WITHOUT_ATTRIBUTE.add("Greater")
    NODE_WITHOUT_ATTRIBUTE.add("Equal")
    NODE_WITHOUT_ATTRIBUTE.add("NonZero")
    NODE_WITHOUT_ATTRIBUTE.add("Relu")
    NODE_WITHOUT_ATTRIBUTE.add("Sigmoid")
    NODE_WITHOUT_ATTRIBUTE.add("Slice")

def init_node_with_attr():
    NODE_WITH_ATTRIBUTE.add("Conv")
    NODE_WITH_ATTRIBUTE.add("Softmax")
    NODE_WITH_ATTRIBUTE.add("ReduceSum")
    NODE_WITH_ATTRIBUTE.add("Concat")
    NODE_WITH_ATTRIBUTE.add("Transpose")
    NODE_WITH_ATTRIBUTE.add("MaxPool")
    NODE_WITH_ATTRIBUTE.add("AveragePool")
    NODE_WITH_ATTRIBUTE.add("Gemm")
    NODE_WITH_ATTRIBUTE.add("Cast")
    NODE_WITH_ATTRIBUTE.add("Flatten")
    NODE_WITH_ATTRIBUTE.add("Pad")
    NODE_WITH_ATTRIBUTE.add("LeakyRelu")

def get_node_parse_type(op_type):
    if len(NODE_WITHOUT_ATTRIBUTE) == 0 or len(NODE_WITH_ATTRIBUTE) == 0:
        init_node_without_attr()
        init_node_with_attr()
    if op_type in NODE_WITHOUT_ATTRIBUTE and op_type not in NODE_WITH_ATTRIBUTE:
        return "node_without_attr"
    elif op_type not in NODE_WITHOUT_ATTRIBUTE and op_type in NODE_WITH_ATTRIBUTE:
        return "node_with_attr"
    else:
        print("{} cannot both wiht attr and without attr!!!!!".format(op_type))
        return ""

    
def get_node_info(node_type, node_info, weights_info):
    node_parse_type = get_node_parse_type(node_type)
    if node_parse_type != "":
        return NODE_PARSE_FUNC[node_parse_type](node_info, weights_info)
    else:
        print("not support {} op for now!!!!".format(node_type))
        return ""

def parse_onnx_graph(graph, weights_info):
    node  = graph.node
    node_info_map = {}
    net_json_graph = {}
    for i in range(len(node)):
        node_info = get_node_info(node[i].op_type, node[i], weights_info)
        if node_info != "":
            node_info_map[node[i].name] = node_info
    net_json_graph["nodes_info"] = node_info_map
    # net_json_graph["weights_info"] = weights_info
    return net_json_graph

def save_simplify_graph(simply_graph, name):
    json_str = json.dumps(simply_graph)
    with open(name + '_graph.json', 'w') as f:
        f.write(json_str)

def get_graph_weights(graph):
    weights = graph.initializer
    output_tensor = list(graph.output)
    output_tensor = [ x.name for x in output_tensor ]
    input_tensor  = list(graph.input)
    fp16_flag = 0
    offset = 0
    weights_info = {}
    for i in range(len(input_tensor)):
        temp = {}
        temp["count"] = 0
        temp["offset"] = -1
        temp["data_type"] = input_tensor[0].type.tensor_type.elem_type
        dims = input_tensor[i].type.tensor_type.shape.dim
        input_shape = [dims[i].dim_value for i in range(len(dims))]
        temp["tensor_shape"] = input_shape
        weights_info[input_tensor[i].name] = temp
        if temp["data_type"] == onnx_data_type["FLOAT16"]:
            fp16_flag = 1

    for ele in weights:
        temp = {}
        temp["offset"] = offset
        temp["data_type"] = ele.data_type
        shape = list(ele.dims)
        if shape == []:
            shape = [1]
        temp["tensor_shape"] = shape
        count, raw_data = get_node_raw_data(ele)
        temp["count"] = count
        temp["raw_data"] = raw_data
        offset = offset + count
        weights_info[ele.name] = temp

    weights_info["net_output"] = output_tensor
    return weights_info, fp16_flag


def update_weights_offset_and_save(weights_info, file_name):
    offset = 0
    with open(file_name + "_weights.bin", "wb") as f:
        for ele in weights_info:
            if "offset" in weights_info[ele] and weights_info[ele]["offset"] != -1:
                weights_info[ele]["offset"] = offset
                offset += weights_info[ele]["count"]
                f.write(weights_info[ele]["raw_data"])
                del weights_info[ele]["raw_data"]

def topo_lazy_search(visited_tensors, nodes_info, topo_node_order):
    for ele in nodes_info.keys():
        flag = True
        node_info = nodes_info[ele]
        for input_tensor in node_info["inputs"]:
            if input_tensor not in visited_tensors:
                flag = False
        if flag == True and (ele not in topo_node_order):
            topo_node_order.append(ele)
            visited_tensors.extend(node_info["outputs"])


def generate_topo_order(nodes_info, weights_info):
    topo_node_order = []
    input_tensors = []
    visited_tensors = []
    for tensor_name in weights_info.keys():
        if isinstance(weights_info[tensor_name],dict):
            if weights_info[tensor_name]["offset"] == -1:
                input_tensors.append(tensor_name)
            visited_tensors.append(tensor_name)
    
    while len(topo_node_order) < len(nodes_info):
        topo_lazy_search(visited_tensors, nodes_info, topo_node_order)
    return topo_node_order
    
def find_depend_node(graph, out_tensor):
    for ele in graph.keys():
        if ele == "weights_info":
            continue
        node_info = graph[ele]
        if out_tensor in node_info["outputs"]:
            return ele
    return None

def generate_depend_nodes(topo_order, simply_graph):
    topo_order.reverse()
    depend_nodes = {}
    for elem in topo_order:
        input_tensors = simply_graph[elem]["inputs"]
        node = []
        for index in range(len(input_tensors)):
            temp_node = find_depend_node(simply_graph, input_tensors[index])
            if (temp_node not in node) and (temp_node != None):
                node.append(temp_node)
        
        depend_nodes[elem] = node

    topo_order.reverse()
    return depend_nodes



if __name__ == "__main__":
    
    #-------------- 1 load onnx model ---------------------
    # onnx_file_name = "/home/xj-zjd/桌面/qiantai_map/test_onnxruntime/tensorrt_wrapper_for_onnx/example/lenet/lenet_simplify.onnx"
    onnx_file_name = sys.argv[1]
    onnx_model = onnx.load(onnx_file_name)
    json_file_prefix = os.path.dirname(onnx_file_name) + "/net"
    weight_file_prefix = json_file_prefix

    #-------------- 2 check onnx model -------------- 
    # onnx.checker.check_model(onnx_model)

    #-------------- 3 parse onnx graph -------------- 
    graph = onnx_model.graph
    weights_info, fp16_flag = get_graph_weights(graph)
    simply_graph = parse_onnx_graph(graph, weights_info)

    #-------------- 4 update weights info offset and save to file -------------- 
    update_weights_offset_and_save(weights_info, weight_file_prefix)

    #-------------- 5 generate topology order from simply graph -------------- 
    topo_order = generate_topo_order(simply_graph["nodes_info"], weights_info)

    #-------------- 6 save weights and simplify graph to files used for tensorrt -------------- 
    simply_graph["topo_order"] = topo_order
    simply_graph["weights_info"] = weights_info
    simply_graph["fp16_flag"] = fp16_flag
    save_simplify_graph(simply_graph, json_file_prefix)

    print("convert success!!!")