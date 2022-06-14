import os
import numpy as np
from cuda import cudart
import tensorrt as trt

class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibrationCount, calibFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibrationCount = calibrationCount
        self.calibFile = np.load(calibFile)

        self.speech_data = self.calibFile['speech-256']
        self.speech_lengths_data = self.calibFile['speech_lengths-256']

        self.speechBuffeSize = trt.volume(self.speech_data.shape) * trt.float32.itemsize
        self.speech_lengthsBuffeSize = trt.volume(self.speech_lengths_data.shape) * trt.float32.itemsize
        _, self.dIn_speech = cudart.cudaMalloc(self.speechBuffeSize)
        _, self.dIn_speech_lengths = cudart.cudaMalloc(self.speech_lengthsBuffeSize)
        self.count = 0
        self.cacheFile = './int8.cache'

    def __del__(self):
        cudart.cudaFree(self.dIn_speech)
        cudart.cudaFree(self.dIn_speech_lengths)

    def get_batch_size(self):  # do NOT change name
        return self.speech_data.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        print(nameList)
        print(inputNodeName)
        if self.count < self.calibrationCount:
            self.count += 1
            cudart.cudaMemcpy(self.dIn_speech, self.speech_data.ctypes.data, self.speechBuffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cudart.cudaMemcpy(self.dIn_speech_lengths, self.speech_lengths_data.ctypes.data, self.speech_lengthsBuffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn_speech), int(self.dIn_speech_lengths)]
        else:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator(5, (1, 1, 28, 28), "./int8.cache")
    m.get_batch("FakeNameList")
