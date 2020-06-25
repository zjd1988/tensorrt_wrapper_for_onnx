import onnx
import onnxmltools
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('../example/shufflenet/shufflenet.onnx')


optimizers_list = ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                        'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                        'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                        'eliminate_nop_transpose', 'fuse_add_bias_into_conv', 
                                        'fuse_consecutive_log_softmax',
                                        'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                                        'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                                        'fuse_pad_into_conv', 'fuse_transpose_into_gemm']

optimizers_list.append('fuse_bn_into_conv')

check_result = onnx.checker.check_model(model)
for i in range(len(model.graph.node)):
    print(model.graph.node[i].input)
# convert model
model_simp = onnx.optimizer.optimize(model, optimizers_list)

onnxmltools.utils.save_model(model_simp, '../example/shufflenet/shufflenet_simplify.onnx')
