import onnxruntime
import numpy as np
import time
devices = onnxruntime.get_device()
session = onnxruntime.InferenceSession("../example/lenet/lenet.onnx")
session.get_modelmeta()
first_input_name = session.get_inputs()[0].name

indata1 = np.ones((1,1,32,32)).astype(np.float32)
results = session.run([], {first_input_name : indata1})

starttime = time.time()
for i in range(1):
    results = session.run([], {first_input_name : indata1})

endtime = time.time()
print((endtime - starttime))
print(results[0])

