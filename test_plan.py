import ctypes
import numpy as np
np.random.seed(0)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import json
import base64
import io

class MyProfiler(trt.IProfiler):
    def __init__(self):
        super(MyProfiler, self).__init__()

    def report_layer_time(self, layerName, ms):
        print("Timing: %8.3fus -> %s"%(ms*1000,layerName))

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for idx, binding in enumerate(engine):
        size = trt.volume(engine.get_binding_shape(binding)) * 1
        print(size)
        host_mem = cuda.pagelocked_empty(size, dtype=trt.nptype(engine.get_binding_dtype(idx)))
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    
    return inputs, outputs, bindings, stream

def trt_inference():
    engine_file = "./test.plan"
    logger = trt.Logger(trt.Logger.WARNING)

    with trt.Runtime(logger) as trt_runtime:
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_file, 'rb') as f:
            engine_data = f.read()

    with open("output.txt", "r") as read_file:
        decodedArray = json.load(read_file)
        json_file = decodedArray['lst'][0][1][0]['outputs']

        def decode(array):

            data = base64.b64decode(array.encode(), validate=True)
            infile = io.BytesIO(data)
            return np.load(infile, allow_pickle=False).astype(np.float32)

        x = decode(json_file['646']['values']['array'])
        mask = decode(json_file['613']['values']['array'])
        pos_emb = decode(json_file['603']['values']['array'])            
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        with engine.create_execution_context() as context:
            context.profiler = MyProfiler()
            np.copyto(inputs[0].host, x.ravel())
            np.copyto(inputs[1].host, mask.ravel())
            np.copyto(inputs[2].host, pos_emb.ravel())

            for inp in inputs:
                cuda.memcpy_htod_async(inp.device, inp.host, stream)
            stream.synchronize()
            context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
            for out in outputs:
                cuda.memcpy_dtoh_async(out.host, out.device, stream) 
                
            stream.synchronize()

            trt_output = [out.host for out in outputs]
            print(trt_output)
            np.save('ouput.npy', trt_output[0])
ctypes.cdll.LoadLibrary("./mha/build/libMHAPlugin.so")
trt_inference()
