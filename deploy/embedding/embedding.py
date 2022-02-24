import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time
from collections import namedtuple, OrderedDict
import torch
import sys
sys.path.append("..")
from trt_inference import TrtEngine


class Embedding(TrtEngine):
    def __init__(self, trt_file=None, gpu_idx=0, batch_size=10):
        super(Embedding, self).__init__(trt_file, gpu_idx, batch_size)

    def inference(self, imgs, new_size=(256, 128)):
        trt_inputs = []
        for img in imgs:
            input_ndarray = self.preprocess(img, *new_size)
            trt_inputs.append(input_ndarray)
        trt_inputs = np.vstack(trt_inputs)

        valid_bsz = trt_inputs.shape[0]
        if valid_bsz < self._batch_size:
            trt_inputs = np.vstack([trt_inputs, np.zeros((self._batch_size - valid_bsz, 3, *new_size))])

        result = self.infer(trt_inputs)[0]
        result = result[:valid_bsz]
        feat = self.postprocess(result, axis=1)
        return feat

    def __call__(self, image, bbx):
        if len(bbx) == 0:
            return []
        imgs = []
        for b in bbx:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            imgs.append(image[y1:y2, x1:x2, :])
        return self.inference(imgs)

    @classmethod
    def preprocess(cls, img, img_height, img_width):
        # Apply pre-processing to image.
        resize_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        type_img = resize_img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        return type_img

    @classmethod
    def postprocess(cls, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def __del__(self):
        del self._input
        del self._output
        del self._stream
        self._device_ctx.detach()  # release device context


if __name__ == "__main__":
    model_wb = Embedding(trt_file="weights/whole_body.engine")
    img0 = cv2.imread('0.jpg')
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    # feat0 = model_wb(img0, [[0, 0, img0.shape[1], img0.shape[0]]])
    # feat1 = model_wb(img1, [[0, 0, img1.shape[1], img1.shape[0]]])
    # feat2 = model_wb(img2, [[0, 0, img2.shape[1], img2.shape[0]]])
    feat0 = model_wb.inference([img0])
    feat1 = model_wb.inference([img1])
    feat2 = model_wb.inference([img2])
    print(feat0)
    print("same id similarity: ", np.dot(feat0, feat1.T))
    print("different id similarity: ", np.dot(feat0, feat2.T))
