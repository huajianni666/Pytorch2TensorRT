import sys
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import pydot
import torchvision
import os
from graphviz import Digraph
from PIL import Image
import numpy as np
import pdb
import pycuda.driver as cuda
import pycuda.autoinit
import common
import tensorrt as trt
from trt_plugin import pluginOp
from common import do_inference, allocate_buffers
import cv2
from torchvision import transforms as T
from torch.autograd import Variable

def plot_graph(top_var, fname, params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.
    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    graph.write_png(im_name)
    print(im_name)

    return im_name

def checkType(vars, types):
    for var,t in zip(vars, types):
        if not isinstance(var ,t):
            return False
    return True

def checkShape(vars, expected):
    for var, e in zip(vars, expected):
        if var.shape != e:
            return False
    return True

class CaffePoolingFormula(trt.IOutputDimensionsFormula):
    def __init__(self):
        pass
    def compute(self, input_shape, kernel_shape, stride, padding, dilation, layer_name):
        hw = []
        for i in range(2):
            out = math.ceil((input_shape[i] + 2 * padding[i] - kernel_shape[i]) / stride[i] + 1)
            hw.append(out)
            print(out)
        return trt.DimsHW(hw)

class ConvLayer:
    def __call__(self, network, inputs, outputs, params):
        """
        添加卷积层
        当bias=False时, len(inputs) == 2, 此时bias = np.zeros(output_channels, dtype=np.float32)
        Args:
            inputs: input, weight, bias -> ITensor, np.ndarray, np.ndarray
            outputs: [(int, tuple),...] 前者对应output的uniquename，后者是output的shape
            params:
        Returns:
            ITensor: trt的卷积输出
        示例:
            %137 : Float(1, 64, 300, 300) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]](%0, %2, %3), scope: SSD/Sequential/Conv2d[0]
        """
        tensor, weight = inputs[:2]
        output_channels = weight.shape[0]
        if len(inputs) == 2:
            bias = np.zeros(output_channels, dtype=np.float32)
        else:
            bias = inputs[2]

        assert checkType([tensor, weight, bias], [trt.ITensor, np.ndarray, np.ndarray]), "Input Type Error in Conv"

        kernel_shape = params['kernel_shape']

        conv = network.add_convolution(tensor, output_channels, kernel_shape, weight, bias)
        conv.stride = params['strides']
        conv.num_groups = params['group']
        if len(params['pads']) == 4:
            conv.padding = params['pads'][:2]
        else:
            conv.padding = params['pads']
        conv.dilation = params['dilations']

        return conv.get_output(0)

class FullyConnectedLayer:
    def __init__(self, bias = True):
        self.bias = bias
    def __call__(self, network, inputs, outputs, params):
        """
        添加全连接层
        当bias=False时, len(inputs) == 2, 此时bias = np.zeros(output_channels, dtype=np.float32)
        且此时weight会有一次转置操作,所以需要将weight转置
        Args:
            inputs: input, weight, bias -> ITensor, np.ndarray, np.ndarray
            outputs: [(int, tuple),...] 前者对应output的uniquename，后者是output的shape
            params:
        Returns:
            ITensor: trt的卷积输出
        示例:
        """
        tensor, weight = inputs[:2]
        assert checkType([tensor, weight], [trt.ITensor, np.ndarray])
        output_channels = outputs[0][1][1]
        if self.bias:
            bias = inputs[2]
            assert checkType([bias], [np.ndarray])
        else:
            # 不带bias的fc操作会先将weight转置，因此需要转置回去
            weight = weight.T
            bias = np.zeros(output_channels, dtype=np.float32)
        fc = network.add_fully_connected(tensor, output_channels, weight, bias)
        return fc.get_output(0)

class ScaleLayer:
    def genScaleLayer(self, network, input, shift, scale, power):
        """
        生成一个trt的scale layer, output = (input * scale + shift)^power
        Args:
            network: trt.INetworkDefinition
            inputs: [input] -> ITensor
            outputs: [(int, tuple),...]
            shift: np.ndarray
            scale: np.ndarray
            power: np.ndarray
        Returns:
            trt.IScaleLayer: trt的scale layer
        """
        assert checkType([input, shift, scale, power], [trt.ITensor, np.ndarray, np.ndarray, np.ndarray])
        # print(shift.dtype, scale.dtype, power.dtype)
        # print(shift.shape, scale.shape, power.shape, input.shape)
        if checkShape([shift, scale, power], [tuple(input.shape)] * 3):
            mode = trt.ScaleMode.ELEMENTWISE
        elif checkShape([shift, scale, power], [(input.shape[0],)] * 3):
            mode = trt.ScaleMode.CHANNEL
        elif checkShape([shift, scale, power], [(1,)] * 3):
            mode =trt.ScaleMode.UNIFORM
        else:
            assert False, "scale params have wrong shape"
        layer = network.add_scale(input, mode, shift, scale, power)
        return layer

class BatchNormalization(ScaleLayer):
    def __call__(self, network, inputs, outputs, params):
        """
        利用scale layer生成bn层
        Args:
            network: trt.INetworkDefinition
            inputs: input -> ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            ITensor: trt.IScaleLayer的输出
        示例:
            %138 : Float(1, 64, 300, 300) = onnx::BatchNormalization[epsilon=1e-05, is_test=1, momentum=1](%137, %4, %5, %6, %7), scope: SSD/Sequential/BatchNorm2d[1]
        """
        assert len(inputs) == 5, "number of inputs is wrong in batchNorm"
        tensor, weight, bias, mean, var = inputs
        checkType(inputs, [trt.ITensor] + [np.ndarray] * 4), "Type Error"
        eps = params["epsilon"]
        scale = weight / np.sqrt(var + eps)
        shift = - weight * mean / np.sqrt(var + eps) + bias
        power = np.ones_like(scale, dtype=np.float32)
        layer = self.genScaleLayer(network, tensor, shift, scale, power)
        return layer.get_output(0)

class Sqrt(ScaleLayer):
    def __call__(self, network, inputs, outputs, params):
        """
        利用scale layer进行开方操作
        Args:
            network: trt.INetworkDefinition
            inputs: [input] -> ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            ITensor: trt.IScaleLayer的输出
        示例:
            %171 : Float(1, 1, 37, 37) = onnx::Sqrt(%170), scope: SSD/L2Norm
        """
        assert len(inputs) == 1, "Input Error"
        checkType(inputs, [trt.ITensor])
        scale = np.array([1.0],dtype=np.float32)
        shift = np.array([0.0],dtype=np.float32)
        power = np.array([0.5],dtype=np.float32)
        layer = self.genScaleLayer(network, inputs[0], shift, scale, power)
        return layer.get_output(0)


class ActivationLayer:
    def __init__(self, activationType):
        """
        Args:
            activationType: trt.ActivationType
        """
        self.activationType = activationType
    def __call__(self, network, inputs, outputs, params):
        """
        添加激活函数
        Args:
            network: trt.INetworkDefinition
            inputs: [input] -> ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            ITensor: trt激活函数的输出
        示例:
            %186 : Float(1, 512, 37, 37) = onnx::Relu(%185), scope: SSD/Sequential/ReLU[32]
        TODO:
            Parser的func中，暂未添加对于其它激活函数的支持(SIGMOID与TANH)
        """
        tensor = inputs[0]
        checkType([tensor], [trt.ITensor]), "Input Type Error"
        relu = network.add_activation(tensor, self.activationType)
        return relu.get_output(0)

class PoolingLayer:
    def __init__(self, poolType):
        """
        Args:
            poolType: trt.PoolingType
        """
        self.poolType = poolType
    def __call__(self, network, inputs, outputs, params):
        """
        添加激活函数
        Args:
            network: trt.INetworkDefinition
            inputs: [input] -> ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            ITensor: trt池化的输出
        示例:
            %187 : Float(1, 512, 18, 18) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2]](%186), scope: SSD/Sequential/MaxPool2d[33]
        TODO:
            更改pooling_output_dimensions_formula后,BN解析有问题
            trt默认的池化层的输出大小是向下取整， pytorch中池化层中有ceil_mode参数, 当ceil_mode==True时
            需要向上取整，此时需要更改trt_network的pooling_output_dimensions_formula参数
        """
        window_size = params['kernel_shape']
        tensor = inputs[0]
        assert checkType([tensor], [trt.ITensor]), "Input Type Error in Pooling"
        pool = network.add_pooling(tensor, self.poolType, window_size)
        pool.stride = params['strides']
        if len(params['pads']) == 4:
            pool.padding = params['pads'][:2]
        else:
            pool.padding = params['pads']
        return pool.get_output(0)

class AtenPoolingLayer:
    def __init__(self, poolType):
        """
        Args:
            poolType: trt.PoolingType
        """
        self.poolType = poolType
    def __call__(self, network, inputs, outputs, params):
        """
        添加激活函数
        Args:
            network: trt.INetworkDefinition
            inputs: [input] -> ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            ITensor: trt池化的输出
        示例:
            %187 : Float(1, 512, 18, 18) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2]](%186), scope: SSD/Sequential/MaxPool2d[33]
        TODO:
            更改pooling_output_dimensions_formula后,BN解析有问题
            trt默认的池化层的输出大小是向下取整， pytorch中池化层中有ceil_mode参数, 当ceil_mode==True时
            需要向上取整，此时需要更改trt_network的pooling_output_dimensions_formula参数
        """
        window_size = inputs[0].shape
        tensor = inputs[0]
        assert checkType([tensor], [trt.ITensor]), "Input Type Error in Pooling"
        pool = network.add_pooling(tensor, self.poolType, window_size[1:])
        return pool.get_output(0)

class Constant:

    def __call__(self, network, inputs, outputs, params):
        """
        根据params中的value字段生成一个常量，用作之后某些操作的参数
        Args:
            network: trt.INetworkDefinition
            inputs: []
            outputs: [(int, tuple)]
            params: {}
        Returns:
            np.ndarray:
        示例:
            %245 : Long() = onnx::Constant[value={-1}](), scope: SSD -1
        """
        value = params['value']
        if type(value) == torch.Tensor and value.ndimension() == 1:
            return np.array(value)
        else:
            assert False, "Constant Type Error"

class ElementWiseLayer:
    def __init__(self,  op):
        """
        args:
            op: trt.ElementWiseOperation
        """
        self.op = op
    def __call__(self, network, inputs, outputs, params):
        """
        进行张量间的7种运算：加，减，乘，除，幂，最大值，最小值
        当输入的两个张量都为网络某一个层的输出时(trt.ITensor)，两个张量形状必须一致
        当输入的两个张量其中之一为网络的参数时(np.ndarray)，参数张量的形状可以是(1,),(C,),(C,H,W),其中C,H,W是另一个张量的形状
        且此时运算为加减乘除幂时，可以调用trt的scale layer完成该运算
        Args:
            network: trt.INetworkDefinition
            inputs: input1, input2 -> trt.ITensor or np.ndarray, trt.ITensor or np.ndarray
            outputs: [(int, tuple)]
            params: {}
        Returns:
            trt.ITensor :trt.IElementWiseLayer的输出或trt.IScaleLayer的输出
        示例:
            %173 : Float(1, 1, 37, 37) = onnx::Add[broadcast=1](%171, %172), scope: SSD/L2Norm
            %169 : Float(1, 512, 37, 37) = onnx::Pow[broadcast=1](%167, %168), scope: SSD/L2Norm

        """
        assert len(inputs) == 2, "number of inputs is wrong"
        x, y = inputs
        if checkType(inputs, [trt.ITensor, np.ndarray]):
            shift, scale, power = self.scaleHelper(y)
            layer = ScaleLayer().genScaleLayer(network, x, shift, scale, power)
        elif checkType(inputs, [np.ndarray, trt.ITensor]):
            shift, scale, power = self.scaleHelper(x)
            layer = ScaleLayer().genScaleLayer(network, y, shift, scale, power)
        elif checkType(inputs, [trt.ITensor, trt.ITensor]):
            layer = network.add_elementwise(x, y, self.op)
        else:
            assert False, "Type Error"
        return layer.get_output(0)

    def scaleHelper(self, y):
        shift = np.zeros_like(y, dtype=np.float32)
        scale = np.ones_like(y, dtype=np.float32)
        power = np.ones_like(y, dtype=np.float32)
        y = y.astype(np.float32)
        if self.op == trt.ElementWiseOperation.SUM:
            shift = y
        elif self.op == trt.ElementWiseOperation.PROD:
            scale = y
        elif self.op == trt.ElementWiseOperation.POW:
            power = y
        elif self.op == trt.ElementWiseOperation.SUB:
            shift = -y
        elif self.op == trt.ElementWiseOperation.DIV:
            scale = 1 / y
        else:
            assert False, "当运算求为最大或最小值时，第二个输入不能是常量"
        return shift, scale, power

class ReduceLayer:
    def __init__(self, reduceType):
        """
        args:
            reduceType: trt.ReduceOperation
        """
        self.reduceType = reduceType
    def __call__(self, network, inputs, outputs, params):
        """
        根据params中的value字段生成一个常量，用作之后某些操作的参数
        Args:
            network: trt.INetworkDefinition
            inputs: input -> trt.ITensor
            outputs: [(int, tuple)]
            params: {}
        Returns:
            trt.ITensor:trt.IReduceLayer的输出
        示例:
        """
        tensor = inputs[0]
        assert checkType([tensor], [trt.ITensor]), "Input Type Error"
        dim = params["axes"][0]
        assert dim >= 1 and dim <= 3
        #Bit 0 corresponds to the C, Bit 1 corresponds to the H, Bit 2 corresponds to the W
        dim = 2**(dim - 1)
        layer = network.add_reduce(tensor, self.reduceType, dim, True)
        return layer.get_output(0)

class ShuffleLayer:
    def __init__(self, transpose = False, reshape = False):
        """
        Args:
            transpose: bool 利用shufflelayer 执行transpose操作
            reshape: bool 利用shufflelayer 执行reshape操作
        """
        self.transpose = transpose
        self.reshape = reshape
        assert self.transpose ^ self.reshape, "Only support one operations one time"
    def __call__(self, network, inputs, outputs, params):
        """
        添加shuffle layer

        Args:
            当transpose=True时:
                inputs: input -> ITensor or np.ndarray
                outputs:[(int,tuple),...]
            当reshape=True时:
                inputs: input, shape -> ITensor or np.ndarray, list[int]
                outputs:[(int,tuple),...]
                params:{}
        Returns:
            ITensor, trt shuffle layer的输出
        说明：
            shuffle layer中维度的计算不包括N(batchSize)，CHW->(0,1,2)
            reshape时: -1表示自动计算，0表示复制维度, -1只能出现一次
            当输入时trt.ITensor时，reshape的shape参数第一个值必须等于N，且trt的shuffle.reshape_dims是长度为3的tuple，不足3的话，需要补1
            transpose操作同样要求N维度不变，且输入到trt时，需要全部减1，因为trt的Channel对应的是0
        """
        tensor = inputs[0]
        # 当输入是np.ndarray时, 执行tensor的reshape和transpose
        if type(tensor) == np.ndarray and self.reshape:
            return tensor.reshape(inputs[1])
        if type(tensor) == np.ndarray and self.transpose:
            return tensor.transpose(params['perm'])

        if self.reshape:
            shuffle = self.reshapeHelper(network, inputs)
        else:
            shuffle = self.permuteHelper(network, inputs, params['perm'])
        return shuffle.get_output(0)

    def reshapeHelper(self, network, inputs):
        if len(inputs) > 1: #reshape
            tensor, shape = inputs
            checkType([tensor, shape], [trt.ITensor, np.ndarray]), "Input Type Error"
            shape = shape.tolist()
            while len(shape) < 4:
                shape.append(1)
            assert len(shape) == 4, "Reshape Error" 
        else: #flatten
            tensor = inputs[0]
            checkType([tensor], [trt.ITensor]), "Input Type Error"
            shape = [0, -1, 1, 1]
            assert len(shape) == 4, "Reshape Error"

        shuffle = network.add_shuffle(tensor)
        shuffle.reshape_dims = shape[1:]
        return shuffle

    def permuteHelper(self, network, inputs, dims):
        tensor = inputs[0]
        checkType([tensor, dims], [trt.ITensor, list]), "Input Type Error"
        assert max(dims) < 4 and dims[0] == 0
        dims = list(map(lambda x: x - 1, dims))[1:]
        shuffle = network.add_shuffle(tensor)
        shuffle.first_transpose = trt.Permutation(dims)
        return shuffle
class SoftmaxLayer:
    def __init__(self, keyword = "dim"):
        self.keyword = keyword
    def __call__(self, network, inputs, outputs, params):
        tensor = inputs[0]
        dim = params[self.keyword]
        assert checkType([tensor, dim], [trt.ITensor, int]), "Input Type Error"
        assert len(outputs) == 1, "Outputs Num Error"
        if dim == -1:
            dim = len(outputs[0][1]) - 1
        assert dim > 0 and dim < 4, "dim wrong in softmax"
        dim = 1 << (dim - 1)
        layer = network.add_softmax(tensor)
        layer.axes = dim
        return layer.get_output(0)

class Concatenation:
    def __call__(self, network, inputs, outputs, params):
        """
        添加concat layer, 对应
        Args:
            inputs : inputs -> List[ITensor] or List[np.ndarray] 需要concat的tensor
            outputs: list[dict] 每一个输出对应一个dict，dict = {'index':int, 'type':str, 'shape':tuple(int)}
            params: dict{'dim': int}
        Returns:
            trt Concatenation后的结果或者np.ndarray拼接后的结果
        注意事项：
            输入如果是ITensor,则在trt中添加concatenation，否则则利用np.concatenate拼接
            对于计算维度时，trt的拼接不考虑N(batchsize）维度，所以pytorch的维度需要减去1才是trt拼接的维度
            暂时没有遇到输入是np.ndarray的情况
        TODO:
            因为当前pytorch代码中priorbox返回的shape=(num,4),而trt的priorbox的shape=(2,num*4),因此需要拼接的维度不一致，之后会修改pytorch代码，并且不允许dim=0的情况

        """
        tensors = inputs
        assert  checkType(tensors, [type(tensors[0])] * len(tensors)), "Input Type Error"
        dim = params['axis']

        if type(tensors[0]) == np.ndarray:
            return np.concatenate(tensors,axis=dim)
        else:
            assert dim < 4
            concatLayer = network.add_concatenation(tensors)
            if dim > 0:
                concatLayer.axis = dim - 1
            #TODO 暂时留做priorbox的特殊情况，之后会禁止dim = 0
            elif dim == 0:
                concatLayer.axis = 1
            return concatLayer.get_output(0)

class DoNothing:
    def __init__(self, optype):
        self.optype = optype
    def __call__(self, network, inputs, outputs, params):
        return inputs[0]


class Slice:
    def __call__(self, network, inputs, outputs, params):
        axes, starts, ends = params['axes'], params['starts'], params['ends']
        tensor = inputs[0]
        checkType(tensor, [np.ndarray]), "Input Type Error"
        y = tensor
        #TODO 假设axes是从零开始，后续需要完善
        for s,e in zip(starts, ends):
            y = y[s:e]
        return y

class Shape:
    def __call__(self, network, inputs, outputs, params):
        tensor = inputs[0]
        y = np.array(tensor.shape)
        return y

class PytorchParser:
    def __init__(self, pluginOp = pluginOp):
        self.funcs = {"onnx::Conv":                 ConvLayer(),
                      "onnx::Gemm":                 FullyConnectedLayer(bias=True),
                      "onnx::MatMul":               FullyConnectedLayer(bias=False),
                      "onnx::BatchNormalization":   BatchNormalization(),
                      "onnx::Sqrt":                 Sqrt(),
                      "onnx::Relu":                 ActivationLayer(trt.ActivationType.RELU),
                      "onnx::MaxPool":              PoolingLayer(trt.PoolingType.MAX),
                      "onnx::AveragePool":          PoolingLayer(trt.PoolingType.AVERAGE),
                      "aten::adaptive_avg_pool2d":  AtenPoolingLayer(trt.PoolingType.AVERAGE),
                      "onnx::Constant":             Constant(),
                      "onnx::Add":                  ElementWiseLayer(trt.ElementWiseOperation.SUM),
                      "onnx::Div":                  ElementWiseLayer(trt.ElementWiseOperation.DIV),
                      "onnx::Mul":                  ElementWiseLayer(trt.ElementWiseOperation.PROD),
                      "onnx::Pow":                  ElementWiseLayer(trt.ElementWiseOperation.POW),
                      "onnx::ReduceSum":            ReduceLayer(trt.ReduceOperation.SUM),
                      "onnx::Softmax":              SoftmaxLayer(keyword="axis"),
                      "aten::softmax":              SoftmaxLayer(keyword="dim"),
                      "onnx::Reshape":              ShuffleLayer(reshape=True),
                      "onnx::Transpose":            ShuffleLayer(transpose=True),
                      "onnx::Concat":               Concatenation(),
                      "onnx::Slice":                Slice(),
                      "onnx::Shape":                Shape(),
                      "onnx::Unsqueeze":            DoNothing(optype="onnx::Unsqueeze"),
                      "onnx::Squeeze":              DoNothing(optype="onnx::Squeeze"),
                      "onnx::Dropout":              DoNothing(optype="onnx::Dropout"),
                      "onnx::Pad":                  DoNothing(optype="onnx::Pad"),
                      "aten::expand_as":            DoNothing(optype="aten::expand_as"),
                      "onnx::Flatten":              ShuffleLayer(reshape=True),
                      }

        self.pluginOp = pluginOp
        self.seen = set()

    def addInput(self, network, inputs, input_names):
        """
        根据pytorch的输入Tensor，生成trt.ITensor并在trt的network中标记
        Args:
            input_vars: list[Tensor]，默认Tensor是NCHW四维
            input_names: list[str], 对应Tensor的name
        Returns:
            inputTensors: list[trt.ITensor], 对应输入的Tensor
        """
        inputTensors = []
        if type(inputs) != tuple:
            inputs, input_names = [inputs,], [input_names,]
        assert len(inputs) == len(input_names)
        idx = 0
        for var, name in zip(inputs, input_names):
            if idx ==0:
                assert var.dim() == 4,'wrong input dim'
                tensor = network.add_input(name, trt.float32, var.shape[1:])
                mean = np.asarray([123, 117, 104], dtype=np.float32)
                scale = np.asarray([0.017, 0.017, 0.017], dtype=np.float32)
                layer = network.add_scale(tensor, trt.ScaleMode.CHANNEL, shift=-scale*mean, scale=scale)
                inputTensors.append(layer.get_output(0))
                idx += 1
            else:
                var = var.detach().numpy()
                layer = network.add_constant(trt.DimsCHW(var.size, 1, 1), var)
                # tensor = network.add_input(name, trt.float32, var.shape[0:])
                inputTensors.append(layer.get_output(0))
        return inputTensors

    def RefinedetParse(self, graph, params, trt_network, input_var, input_names, Vehicle):
        inputTensors = self.addInput(trt_network, input_var, input_names)
        imgdata = []
        imgdata.append(inputTensors[0])
        data = {str(key): val for key, val in enumerate(imgdata + params)}
        for index, node in enumerate(graph.nodes()):
            layerType = node.kind()
            inputs = [data[i.uniqueName()] for i in node.inputs()]
            outputs = [(i.uniqueName(), i.type().sizes() if i.type().kind() == 'TensorType' else []) for i in node.outputs()]
            try:
                if layerType in self.funcs.keys():
                    print('normal op: {},{}'.format(outputs[0][0],layerType))
                    params = {k : node[k] for k in node.attributeNames()}
                    results = self.funcs[layerType](trt_network, inputs, outputs, params)
                else:
                    print('plugin op: {},{}'.format(outputs[0][0],layerType))
                    params = [] #node.scalar_args()
                    results = self.pluginOp[layerType](trt_network, inputs, outputs, params)
            except:
                assert False, "Error in {}".format(str(node).strip())

            if type(results) != list:
                data[outputs[0][0]] = results
            else:
                assert len(results) == len(outputs)
                for i,result in enumerate(results):
                    data[outputs[i][0]] = results[i]
        onnx_output_idx = []
        for node in graph.outputs():
            index = node.uniqueName()
            assert type(data[index]) == trt.ITensor, "output tensor is not a trt.ITensor"
            onnx_output_idx.append(index)

        inputs = [data[onnx_output_idx[0]], inputTensors[1]]
        prior_data  = self.pluginOp["FinetuneLocF"](trt_network, inputs, outputs, params)
        inputs = [data[onnx_output_idx[1]], data[onnx_output_idx[3]]]
        outputsConf = self.pluginOp["FilterBgConfF"](trt_network, inputs, outputs, params)
        params = [Vehicle['NUM_CLASSES'],Vehicle['BG_LABEL'],Vehicle['TOP_K'],Vehicle['CONF_THRESH'],Vehicle['NMS_THRESH']]
        inputs = [data[onnx_output_idx[2]], outputsConf, prior_data]
        result = self.pluginOp["DetectF"](trt_network, inputs, outputs, params)
        trt_network.mark_output(result)
       

def inference(network, engine, input_vars):
    input_vars = [ input_vars[0]]
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        for i, input_var in enumerate(input_vars):
            np.copyto(inputs[i].host, input_var.numpy().reshape(-1))
        do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    return outputs


def build_transform(SIZE = 224, PIX_MEAN = [102.9801, 115.9465, 122.7717], PIX_STD = [1., 1., 1.], TO_BGR255 = True):
    """
    Creates a basic transformation that was used to train the models
    """
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=PIX_MEAN, std=PIX_STD
    )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((SIZE,SIZE)),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


