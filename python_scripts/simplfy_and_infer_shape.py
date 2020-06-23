import onnx
import onnxmltools
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('../example/lenet/lenet.onnx')

# convert model
model_with_shape = onnx.shape_inference.infer_shapes(model)
onnxmltools.utils.save_model(model_with_shape, '../example/lenet/lenet_infer_shape.onnx')

model_simp, check = simplify(model)
onnxmltools.utils.save_model(model_simp, '../example/lenet/lenet_simplify.onnx')
# assert check, "Simplified ONNX model could not be validated"