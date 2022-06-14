import os
import ctypes
import numpy as np
import tensorrt as trt
from cuda import cudart
from glob import glob

cudart.cudaDeviceSynchronize()

onnxFile = "./ModifiedDecoder.onnx"
engineFile = '/target/decoder.plan'

soFileList = glob("/target/*.so")

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')
if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

profile.set_shape("encoder_out", (1,16,256), (64,16,256), (64,256,256))
profile.set_shape("encoder_out_lens", (1,), (64,), (64,))
profile.set_shape("hyps_pad_sos_eos", (1,10,64), (64,10,64), (64,10,64))
profile.set_shape("hyps_lens_sos", (1,10), (64,10), (64,10))
profile.set_shape("ctc_score", (1,10), (64,10), (64,10))

config.add_optimization_profile(profile)
config.max_workspace_size = 24 << 30
config.flags = 1 << int(trt.BuilderFlag.FP16) # default value for A30 128 for TF32, 0 for FP16
print(config.flags)

parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
    for error in range(parser.num_errors):
        print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")


for i in range(network.num_layers):
    layer_type = network.get_layer(i).type
    if layer_type == trt.LayerType.ELEMENTWISE:
        network.get_layer(i).precision = trt.DataType.FLOAT

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

with open(engineFile, 'wb') as f:
    f.write(engine.serialize())

print('convert done')
