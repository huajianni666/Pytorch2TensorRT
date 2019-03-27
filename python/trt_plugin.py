import sys
import tensorrt as trt
sys.path.insert(0, 'plugin/build/')
from plugin import *
import pdb
class PriorboxOp:
    def __call__(self, network, inputs, outputs, params):
        assert type(params) == list and len(params) == 8, "priorbox params error"
        rightTypes = [int, int, list, bool, bool, list, int, float]
        for param, rightType in zip(params, rightTypes):
            assert type(param) == rightType
        minSize, maxSize, aspectRatios, flip, clip, variance, step, offset = params
        if len(variance) == 2:
            variance = [variance[0], variance[0], variance[1], variance[1]]
        assert (len(variance) == 4)
        minSize = [minSize]
        if maxSize:
            maxSize = [maxSize]
        else:
            maxSize = []

        assert len(inputs) == 2
        feature, data = inputs

        assert type(feature) == trt.ITensor and type(data) == trt.ITensor
        plugin = PriorBoxPlugin(minSize, maxSize, aspectRatios, len(minSize), len(maxSize), len(aspectRatios),
                                         flip, clip, variance, 0, 0, step, step, offset)
        layer = network.add_plugin([feature, data], plugin)
        layer.name = 'priorbox'
        #print("priorbox: ", layer.get_output(0).shape, feature.shape, data.shape, minSize)
        #print(minSize, maxSize, aspectRatios, len(minSize), len(maxSize), len(aspectRatios), flip, clip, variance, 0, 0,
        #      step, step, offset)
        return layer.get_output(0)

class DetecOp:
    def __call__(self, network, inputs, outputs, params):
        assert len(inputs) == 3
        loc, conf, prior = inputs
        assert type(loc) == trt.ITensor and type(conf) == trt.ITensor and type(prior) == trt.ITensor
        shareLocation, varianceEncodedInTarget = True, False
        num_classes, backgroundLabelId, top_k, conf_thresh, nms_thresh = params
        keepTopK = top_k
        shuffle1 = network.add_shuffle(loc)
        shuffle1.reshape_dims = [-1, 1, 1]
        loc = shuffle1.get_output(0)
        shuffle2 = network.add_shuffle(conf)
        shuffle2.reshape_dims = [-1, 1, 1]
        conf = shuffle2.get_output(0)
        shuffle3 = network.add_shuffle(prior)
        shuffle3.reshape_dims = [1, -1, 2]
        prior  = shuffle3.get_output(0)
        plugin = DetectionOutputPlugin(shareLocation, varianceEncodedInTarget, backgroundLabelId, num_classes,
                                                top_k, keepTopK, conf_thresh, nms_thresh)

        layer = network.add_plugin([loc, conf, prior], plugin)
        layer.name = 'detectionoutput'
        return layer.get_output(0)

upsamplename = 1
class UpsampleOp:
    def __call__(self, network, inputs, outputs, params):
        assert len(inputs) == 1
        assert type(inputs[0]) == trt.ITensor
        plugin = UpSamplePlugin(2.0, 0)
        layer = network.add_plugin_ext(inputs, plugin)
        global upsamplename
        layer.name = 'upsample'+ str(upsamplename)
        upsamplename += 1
        return layer.get_output(0)

class FilterBgConfOp:
    def __call__(self, network, inputs, outputs, params):
        assert len(inputs) == 2
        assert type(inputs[0]) == trt.ITensor and type(inputs[1]) == trt.ITensor
        plugin = FilterBgConfPlugin(0.01)
        layer = network.add_plugin_ext(inputs, plugin)
        layer.name = 'filterbgconf'
        return layer.get_output(0)

class FinetuneLocOp:
    def __call__(self, network, inputs, outputs, params):
        assert len(inputs) == 2
        assert type(inputs[0]) == trt.ITensor and type(inputs[1]) == trt.ITensor
        plugin = FinetuneLocPlugin()
        layer = network.add_plugin_ext(inputs, plugin)
        layer.name = 'finetuneloc'
        return layer.get_output(0)

pluginOp = {"PriorBoxF":       PriorboxOp(),
            "DetectF":         DetecOp(),
            "FilterBgConfF":   FilterBgConfOp(),
            "FinetuneLocF":    FinetuneLocOp(),
            "onnx::Upsample":  UpsampleOp()
           }
