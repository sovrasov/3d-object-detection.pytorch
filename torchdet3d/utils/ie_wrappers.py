import cv2 as cv
import glog as log
import os
import numpy as np


__all__ = ['Regressor', 'Detector']

class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs_info, input_key, output_key):
        self.net = exec_net
        self.inputs_info = inputs_info
        self.input_key = input_key
        self.output_key = output_key
        self.reqs_ids = []

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
        return list(res.values())

    def forward_async(self, img):
        id_ = len(self.reqs_ids)
        self.net.start_async(request_id=id_,
                             inputs={self.input_key: self._preprocess(img)})
        self.reqs_ids.append(id_)

    def grab_all_async(self):
        outputs = []
        for id_ in self.reqs_ids:
            self.net.requests[id_].wait(-1)
            output_list = [self.net.requests[id_].output_blobs[key].buffer for key in self.output_key]
            outputs.append(output_list)
        self.reqs_ids = []
        return outputs

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key].input_data.shape


def load_ie_model(ie, model_xml, device, plugin_dir, cpu_extension='', num_reqs=1):
    """Loads a model in the Inference Engine format"""
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing Inference Engine plugin for %s ", device)

    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, 'CPU')
    # Read IR
    log.info("Loading network")
    net = ie.read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = net.outputs.keys()
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device, num_requests=num_reqs)
    model = IEModel(exec_net, net.input_info, input_blob, out_blob)
    return model


class Detector:
    """Wrapper class for object detector"""
    def __init__(self, ie,  model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1., 1.)

    def run_async(self, frame):
        self.frame_shape = frame.shape
        self.net.forward_async(frame)

    def wait_and_grab(self):
        outputs = self.net.grab_all_async()
        detections = []
        assert len(outputs) == 1
        detections = self.__decode_detections(outputs[0][0], self.frame_shape)
        return detections

    def get_detections(self, frame):
        """Returns all detections on frame"""
        out = self.net.forward(frame)
        detections = self.__decode_detections(out[0], frame.shape)
        return detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            label = detection[1]
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append((left, top, right, bottom, confidence, label))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)
        return detections


class Regressor:
    """Wrapper class for regression model"""
    def __init__(self, ie,  model_path, device='CPU', ext_path=''):
        self.net = load_ie_model(ie, model_path, device, None, ext_path)

    def get_detections(self, frame, detections):
        """Returns all detections on frame"""
        outputs = []
        for rect in detections:
            cropped_img = self.crop(frame, rect[:4])
            out = self.net.forward(cropped_img)
            out = self.__decode_detections(out, rect)
            outputs.append(out)
        return outputs

    def __decode_detections(self, out, rect):
        """Decodes raw regression model output"""
        label = np.argmax(out[0])
        kp = out[1][label]
        return (kp, label)

    @staticmethod
    def transform_kp(kp: np.array, crop_cords: tuple):
        x0,y0,x1,y1 = crop_cords
        crop_shape = (x1-x0,y1-y0)
        kp[:,0] = kp[:,0]*crop_shape[0]
        kp[:,1] = kp[:,1]*crop_shape[1]
        kp[:,0] += x0
        kp[:,1] += y0
        return kp

    @staticmethod
    def crop(frame, rect):
        x0, y0, x1, y1 = rect
        crop = frame[y0:y1, x0:x1]
        return crop
