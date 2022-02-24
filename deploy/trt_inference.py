import cv2
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
import pickle
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


class HostDeviceMem(object):
    """ Host and Device Memory Package """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


TRT_LOGGER = trt.Logger()


class TrtEngine:
    def __init__(self, trt_file=None, gpu_idx=0, batch_size=1):
        # cuda.init()
        self._batch_size = batch_size
        self._device_ctx = cuda.Device(gpu_idx).make_context()
        self._engine = self._load_engine(trt_file)
        self._context = self._engine.create_execution_context()
        self._input, self._output, self._output_shape, self._bindings, self._stream = self._allocate_buffers(self._context)

    def _load_engine(self, trt_file):
        """
        Load tensorrt engine.
        :param trt_file:    tensorrt file.
        :return:
            ICudaEngine
        """
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(trt_file, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self, context):
        """
        Allocate device memory space for data.
        :param context:
        :return:
        """
        inputs = []
        outputs = []
        output_shapes = []
        bindings = []
        stream = cuda.Stream()
        for binding in self._engine:
            shape = self._engine.get_binding_shape(binding)
            size = trt.volume(shape) * self._engine.max_batch_size
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_shapes.append(shape)
        return inputs, outputs, output_shapes, bindings, stream

    def infer(self, data):
        """
        Real inference process.
        :param model:   Model objects
        :param data:    Preprocessed data
        :return:
            output
        """
        # Copy data to input memory buffer
        [np.copyto(_inp.host, data.ravel()) for _inp in self._input]
        # Push to device
        self._device_ctx.push()
        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self._input.device, self._input.host, self._stream)
        [cuda.memcpy_htod_async(inp.device, inp.host, self._stream) for inp in self._input]
        # Run inference.
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        # Transfer predictions back from the GPU.
        # cuda.memcpy_dtoh_async(self._output.host, self._output.device, self._stream)
        [cuda.memcpy_dtoh_async(out.host, out.device, self._stream) for out in self._output]
        # Synchronize the stream
        self._stream.synchronize()
        # Pop the device
        self._device_ctx.pop()

        # return [out.host.reshape(self._batch_size, *shape) for out, shape in zip(self._output[::-1], self._output_shape[::-1])]
        return [out.host.reshape(*shape) for out, shape in zip(self._output[::-1], self._output_shape[::-1])]

    def __del__(self):
        del self._input
        del self._output
        del self._stream
        self._device_ctx.detach()  # release device context
