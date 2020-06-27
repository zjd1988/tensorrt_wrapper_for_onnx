import numpy as np
import onnx
import onnxmltools
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('./yolov5l.onnx')
graph = model.graph

remove_index = {}
for i in range(len(graph.node)):
    if graph.node[i].name == "Resize_243" or graph.node[i].name == "Resize_297":
        remove_index[graph.node[i].name] = i
    elif graph.node[i].name == "Reshape_384" or graph.node[i].name == "Reshape_399" or graph.node[i].name == "Reshape_689":
        remove_index[graph.node[i].name] = i
    else:
        continue


#construct new initializer
scale_values = np.random.randn(4).astype(np.float32)
scale_values[0] = 1
scale_values[1] = 1
scale_values[2] = 2
scale_values[3] = 2
scale = onnx.helper.make_tensor('scale_values', onnx.TensorProto.FLOAT, [4], scale_values.tostring(), True)
graph.initializer.append(scale)

reshape_values1 = np.random.randn(5).astype(np.int64)
reshape_values1[0] = 1
reshape_values1[1] = 3
reshape_values1[2] = 85
reshape_values1[3] = 20
reshape_values1[4] = 20
reshape1 = onnx.helper.make_tensor('reshape_values1', onnx.TensorProto.INT64, [5], reshape_values1.tostring(), True)
graph.initializer.append(reshape1)

reshape_values2 = np.random.randn(5).astype(np.int64)
reshape_values2[0] = 1
reshape_values2[1] = 3
reshape_values2[2] = 85
reshape_values2[3] = 40
reshape_values2[4] = 40
reshape2 = onnx.helper.make_tensor('reshape_values2', onnx.TensorProto.INT64, [5], reshape_values2.tostring(), True)
graph.initializer.append(reshape2)

reshape_values3 = np.random.randn(5).astype(np.int64)
reshape_values3[0] = 1
reshape_values3[1] = 3
reshape_values3[2] = 85
reshape_values3[3] = 80
reshape_values3[4] = 80
reshape3 = onnx.helper.make_tensor('reshape_values3', onnx.TensorProto.INT64, [5], reshape_values3.tostring(), True)
graph.initializer.append(reshape3)


#resize scale node and reshape node to remove
for name,index in remove_index.items():
    if name == "Resize_243" or name == "Resize_297":
        graph.node[index].input.pop()
        graph.node[index].input.append("scale_values")
    elif name == "Reshape_384":
        graph.node[index].input.pop()
        graph.node[index].input.append("reshape_values1")
    elif name == "Reshape_399":
        graph.node[index].input.pop()
        graph.node[index].input.append("reshape_values2")  
    elif name == "Reshape_689":
        graph.node[index].input.pop()
        graph.node[index].input.append("reshape_values3")
    else:
        pass                      

model_simp, check = simplify(model)
onnxmltools.utils.save_model(model_simp, './yolov5l_simplify.onnx')