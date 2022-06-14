import onnx_graphsurgeon as gs
import numpy as np
import onnx

model = onnx.load("/workspace/encoder.onnx")
graph = gs.import_onnx(model)

tensors = graph.tensors()

graph.inputs = [tensors["646"].to_variable(dtype=np.float32, shape=(64, 63, 256))]
graph.outputs = [tensors["652"].to_variable(dtype=np.float32)]

graph.cleanup()

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
