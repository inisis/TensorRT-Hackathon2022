import onnx
import onnxruntime as rt
import numpy as np


onnx_rt_dict = {'646': np.load('input.npy').astype(np.float32)}

sess = rt.InferenceSession('subgraph.onnx')
onnx_outname = [output.name for output in sess.get_outputs()]
res = sess.run(onnx_outname, onnx_rt_dict)


output = np.load('ouput.npy').flatten()

print(res[0].flatten())
print((abs(res[0].flatten()-output)>0.0001).sum())
