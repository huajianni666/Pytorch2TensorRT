import pdb
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import tensorrt as trt
import pycuda
import pycuda.autoinit

from pytorchParse import PytorchParser, build_transform
from common import do_inference, allocate_buffers


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def inference(network, builder, input_vars):
    if type(input_vars) != tuple:
        input_vars = [input_vars]
    builder.max_workspace_size = 256 << 20
    engine = builder.build_cuda_engine(network)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    for i, input_var in enumerate(input_vars):
        np.copyto(inputs[i].host, input_var.numpy().reshape(-1))
    with engine.create_execution_context() as context:
        do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    return outputs

def compare(net, input_vars, trt_outputs):
    net.eval()
    if type(input_vars) != tuple:
        input_vars = [input_vars]
    with torch.no_grad():
        torch_outputs = net.forward(*input_vars)
    if type(torch_outputs) == torch.Tensor:
        torch_outputs = [torch_outputs]
    for i in range(len(trt_outputs)):
        y1 = trt_outputs[i].host
        y2 = torch_outputs[i].numpy().reshape(-1)
        print("output {}, max diff: {}".format(i, np.max(np.abs(y1-y2))))

def testModel(torch_network, input_vars, desc = None):
    parser = PytorchParser()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as trt_network:
        if type(input_vars) == tuple:
            names = [str(i) for i in range(len(input_vars))]
        else:
            names = "1"
        parser.parse(torch_network, trt_network, input_vars, names)
        trt_outputs = inference(trt_network, builder, input_vars)
        pdb.set_trace()
        if desc:
            print(desc)
        compare(torch_network, input_vars, trt_outputs)
        print()

def printGraph(torch_network, inputs):
    with torch.onnx.set_training(torch_network, False):
        trace, _ = torch.jit.get_trace_graph(torch_network, inputs)
    # only support torch.__version__ == 0.4.1.0
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    graph = trace.graph()
    print(graph)


def testConv():
    conv_with_bias = torch.nn.Conv2d(3,6,kernel_size=3)
    conv_without_bias = torch.nn.Conv2d(3,6,kernel_size=3, bias=False)
    input_vars = torch.randn(1,3,10,10)
    testModel(conv_with_bias, input_vars, "conv with bias:")
    testModel(conv_without_bias, input_vars, "conv without bias:")

def testFc():
    class Fc(torch.nn.Module):
        def __init__(self):
            super(Fc, self).__init__()
            self.fc = torch.nn.Linear(18,3)
        def forward(self, input):
            return self.fc(input.view(input.size(0),-1))

    class Fc2(torch.nn.Module):
        def __init__(self):
            super(Fc2, self).__init__()
            self.fc = torch.nn.Linear(18,3,bias=False)
        def forward(self, input):
            return self.fc(input.view(input.size(0),-1))
    net1 = Fc()
    net2 = Fc2()
    x = torch.randn(1,2,3,3)
    testModel(net1, x, "test fc with bias")
    testModel(net2, x, "Test fc without bias:")



def testBN():
    bn = torch.nn.BatchNorm2d(20).eval()
    input_vars = torch.randn(1,20,5,5)
    testModel(bn, input_vars, "bn:")

def testSqrt():
    class Sqrt(torch.nn.Module):
        def __init__(self):
            super(Sqrt, self).__init__()
        def forward(self, x):
            return x.sqrt()
    net = Sqrt()
    x = torch.rand(1,3,20,20)
    testModel(net, x, "Test Sqrt:")

def testRelu():
    relu = torch.nn.ReLU()
    input_vars = torch.rand(1,20,5,5)
    testModel(relu, input_vars, "relu:")

def testMaxPool():
    max_pool = torch.nn.MaxPool2d(kernel_size=2)
    input_vars = torch.randn(1,20,9,9)
    testModel(max_pool, input_vars)

def testAdd():
    class AddConst(torch.nn.Module):
        def __init__(self):
            super(AddConst, self).__init__()
        def forward(self, x):
            return x + 2
    class AddConst2(torch.nn.Module):
        def __init__(self):
            super(AddConst, self).__init__()
        def forward(self, x):
            return x + torch.Tensor([2,3,4])

    class AddTwo(torch.nn.Module):
        def __init__(self):
            super(AddTwo, self).__init__()
        def forward(self, x, y):
            return x + y

    net1 = AddConst()
    net2 = AddTwo()
    x = torch.randn(1,3,20,20)
    y = torch.randn(1,3,20,20)
    testModel(net1, x, "Add a const number:")
    testModel(net2,(x,y), "Add a tensor:")

def testSquare():
    class Square(torch.nn.Module):
        def __init__(self):
            super(Square, self).__init__()
        def forward(self, x):
            return x.pow(2)
    net = Square()
    x_p = torch.rand(1,3,20,20)
    x_n = torch.randn(1,3,20,20)
    testModel(net, x_p, "Test Square(All number are positive)")
    testModel(net, x_n, "Test Square(Number are negtive and positive,the anwser is wrong)")

def testShuffle():
    class Reshape(torch.nn.Module):
        def __init__(self):
            super(Reshape, self).__init__()
        def forward(self, input):
            return input.view(input.size(0), -1)
    class Transpose(torch.nn.Module):
        def __init__(self):
            super(Transpose, self).__init__()
        def forward(self, input):
            return input.permute(0,2,3,1)
    net1 = Reshape()
    net2 = Transpose()
    x = torch.randn(1,3,20,21)
    testModel(net1, x, "Test Reshape(include size operation):")
    testModel(net2, x, "Test Transpose:")
def testSoftmax():
    #这两个网络解析出来的关键字不一样
    net1 = torch.nn.Softmax(dim=-1)
    net2 = torch.nn.Softmax(dim=2)
    x = torch.randn(1,10,10,10)
    testModel(net1, x, "Test softmax(onnx):")
    testModel(net2, x, "Test softmax(aten):")
def testReduce():
    class Reduce(torch.nn.Module):
        def __init__(self):
            super(Reduce, self).__init__()
        def forward(self, inputs):
            return inputs.sum(dim=1, keepdim=True)
    net = Reduce()
    x = torch.randn(1,10,10,10)
    testModel(net,x,"Test Reduce:")

def testConcat():
    class Concat(torch.nn.Module):
        def __init__(self):
            super(Concat, self).__init__()
        def forward(self, input1, input2):
            return torch.cat([input1, input2], 1)
    net = Concat()
    x = torch.randn(1,10,10,10)
    y = torch.randn(1,20,10,10)
    testModel(net,(x,y), "Test Concat")

def testVGG16():
    from torchvision.models import vgg16_bn
    net = vgg16_bn(pretrained=False).eval()
    x = torch.randn(1,3,224,224)
    testModel(net, x, "Test VGG16:")

def testResnet():
    from torchvision.models import resnet18
    net = resnet18()
    x = torch.randn(1,3,224,224)
    testModel(net, x, desc="Test resnet:")

def testPriorbox():
    from ssd.layers import PriorBox
    from ssd.configures import FACE as cfg

    net = PriorBox(cfg['min_sizes'][0], cfg['max_sizes'][0], cfg['aspect_ratios'][0], cfg['flip'], cfg['clip'],
             cfg['variance'], cfg['steps'][0], cfg['offset'])
    data = torch.randn(1,3,300,300)
    feature = torch.randn(1,512,37,37)

    parser = PytorchParser()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as trt_network:
        parser.parse(net, trt_network, (feature, data), ("feature", "data"))
        trt_outputs = inference(trt_network, builder, (feature, data))

    trt_outputs = trt_outputs[0].host.reshape(2, -1, 4)[0]
    trt_outputs = np.concatenate(((trt_outputs[:,:2] + trt_outputs[:, 2:]) / 2,trt_outputs[:,2:] - trt_outputs[:, :2]),axis=1)

    torch_outputs = net(feature, data)
    print("Test priorbox:")
    print("output {}, max diff: {}".format(0, np.max(
                                    np.abs(trt_outputs.reshape(-1) - torch_outputs.numpy().reshape(-1)))))
    print()


def testSSD(compare = False):
    """
    当ssd中不包含l2norm,priorbox和det时，可以令compare = True比较loc和conf的预测结果
    因为l2norm中pow操作trt对于负数有bug，
    priorbox的python代码产生的shape是(num, 4)且是中心坐标与trt的priorbox结果不能直接比较
    det的python和trt的结果形式同样不一致，也不能直接比较

    当compare = False时，我去除了l2norm，然后打印出了pytorch和trt的前几条结果，发现是一致的
    """
    from ssd.ssd import VGG as BackBone, SSD
    from ssd.configures import FACE

    backbone = BackBone()
    torch_network = SSD(cfg=FACE, backbone=backbone).eval()
    img = cv2.imread('person.jpg')
    transforms = build_transform(SIZE = 300, PIX_MEAN = [104,117,123], PIX_STD = [1., 1., 1.], TO_BGR255 = True)
    input_vars = Variable(transforms(img).unsqueeze(0), volatile=True) 
    if compare:
        testModel(torch_network, input_vars, "Test SSD:")
    else:
        parser = PytorchParser()
        printGraph(torch_network, input_vars)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as trt_network:
            if type(input_vars) == tuple:
                names = [str(i) for i in range(len(input_vars))]
            else:
                names = "1"
            parser.parse(torch_network, trt_network, input_vars, names)
            trt_outputs = inference(trt_network, builder, input_vars)
            print(trt_outputs[0].host.shape)


def testL2Norm():
    from ssd.layers import L2Norm
    net = L2Norm(512,20).eval()
    x = torch.rand(1,512,10,10)
    testModel(net,x,"Test L2 Norm(no negtive number):")



if __name__ == "__main__":
    #testConv()
    #testFc()
    #testBN()
    #testRelu()
    #testMaxPool()
    #testAdd()
    #testSqrt()
    #testSquare()
    #testShuffle()
    #testSoftmax()
    #testReduce()
    #testConcat()
    #testVGG16()
    #testResnet()
    #testL2Norm()
    #testPriorbox()
    testSSD(compare = True)
