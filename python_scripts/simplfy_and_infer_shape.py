import onnx
import onnxmltools
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('hfnet_github_global.onnx')

# convert model
model_with_shape = onnx.shape_inference.infer_shapes(model)
onnxmltools.utils.save_model(model_with_shape, 'hfnet_github_inference.onnx')

model_simp, check = simplify(model)
onnxmltools.utils.save_model(model_simp, 'hfnet_github_simplify.onnx')
# assert check, "Simplified ONNX model could not be validated"