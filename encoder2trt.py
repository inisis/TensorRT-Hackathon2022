import os
import ctypes
import numpy as np
import tensorrt as trt
from cuda import cudart
from glob import glob
# from calibrator import MyCalibrator

cudart.cudaDeviceSynchronize()

onnxFile = "./ModifyEncoder.onnx"
engineFile = '/target/encoder.plan'

soFileList = glob("/target/*.so")
shape = (1, 1, 28, 28)
calibFile = "/workspace/data/calibration.npz"

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
profile.set_shape("speech", (1,16,80), (4,256,80), (64,256,80))
profile.set_shape("speech_lengths", (1,), (4,), (64,))
config.add_optimization_profile(profile)
config.max_workspace_size = 24 << 30
config.flags = 1 << int(trt.BuilderFlag.FP16) # default value for A30 128 for TF32, 0 for FP16
# config.set_flag(trt.BuilderFlag.INT8)
# config.int8_calibrator = MyCalibrator(1, calibFile)

# config.set_flag(trt.BuilderFlag.STRICT_TYPES)
# config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

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

'''
for i in range(network.num_layers):
    layer_type = network.get_layer(i).type
    layer_precision = network.get_layer(i).precision
    layer_name = network.get_layer(i).name

    if layer_type == trt.LayerType.SHAPE or layer_type == trt.LayerType.IDENTITY or layer_type == trt.LayerType.SHUFFLE or layer_type == trt.LayerType.SLICE or layer_type == trt.LayerType.CONCATENATION or layer_type == trt.LayerType.GATHER or layer_type == trt.LayerType.SHUFFLE:
        continue
    if layer_precision == trt.DataType.INT32:
        continue

    if layer_type == trt.LayerType.CONSTANT:
        if network.get_layer(i).get_output_type(0) == trt.DataType.INT32:
            continue
    if layer_type == trt.LayerType.CONVOLUTION or layer_type == trt.LayerType.MATRIX_MULTIPLY or layer_type == trt.LayerType.FULLY_CONNECTED or layer_type == trt.LayerType.ELEMENTWISE:
        continue
    
    network.get_layer(i).precision = trt.DataType.FLOAT
'''

'''
for i in range(network.num_layers):
    layer_type = network.get_layer(i).type
    if layer_type == trt.LayerType.SOFTMAX:
        network.get_layer(i).precision = trt.DataType.FLOAT
'''

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

with open(engineFile, 'wb') as f:
    f.write(engine.serialize())

print('convert done')
