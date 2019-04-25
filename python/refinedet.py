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
import time
import tensorrt as trt
from trt_plugin import pluginOp
from common import do_inference, allocate_buffers
import cv2
from parse import build_transform, inference, PytorchParser, plot_graph, save_engine 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
import torch.onnx
from backbone.model_builder import Refinedet
from backbone.cfg import Vehicle
from backbone.prior_box import getPriorLayer
from backbone.refine_res import RefineResnet18
from collections import OrderedDict

def draw(origimg, rectangles, boxNums): 
    [H,W,C] = origimg.shape
    draw = origimg.copy() 
    font=cv2.FONT_HERSHEY_SIMPLEX
    for i in range(boxNums):
        if rectangles[i * 7 + 1] == -1:
            break
        else:
            if rectangles[i * 7 + 1] == 1:
                cv2.rectangle(draw,(int(rectangles[i*7+3]*W),int(rectangles[i*7+4]*H)),(int(rectangles[i*7+5]*W),int(rectangles[i*7+6]*H)),(255,0,0),2)
            if rectangles[i * 7 + 1] == 2:
                cv2.rectangle(draw,(int(rectangles[i*7+3]*W),int(rectangles[i*7+4]*H)),(int(rectangles[i*7+5]*W),int(rectangles[i*7+6]*H)),(255,255,0),2)
            if rectangles[i * 7 + 1] == 3:
                cv2.rectangle(draw,(int(rectangles[i*7+3]*W),int(rectangles[i*7+4]*H)),(int(rectangles[i*7+5]*W),int(rectangles[i*7+6]*H)),(0,255,0),2)
    cv2.imwrite('result.jpg', draw)

def pre_process(img, resize_wh=[512, 512], swap=(2, 0, 1)):
    interp_methods = [
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
        cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
    ]
    interp_method = interp_methods[0]
    img_info = [img.shape[1], img.shape[0]]
    img = cv2.resize(
        np.array(img), (resize_wh[0], resize_wh[1]),
        interpolation=interp_method).astype(np.float32)
    img = img.transpose(swap)
    return img, img_info

if __name__ == '__main__':
    net = Refinedet(Vehicle,RefineResnet18('448'))
    checkpoint = torch.load("refine_res_epoch_250_300_4_23.pth")
    load_state_dict = checkpoint['model']
    load_keys = sorted(list(load_state_dict.keys()))
    #print('load: {}'.format(load_keys))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in load_state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    net_state_dict = net.state_dict()
    net_keys = sorted(list(net_state_dict.keys()))
    #print('net: {}'.format(net_keys))
    net.load_state_dict(new_state_dict, False)
    net.eval()
    trt_engine_datatype=trt.DataType.FLOAT  #
    img = cv2.imread('test.jpg')
    img_resize, img_info = pre_process(img, [448,448])    
    batch_size = 1
    imgs = []
    imgs.append(img_resize)
    inputs = torch.from_numpy(np.array(imgs))
    #inputs = torch.ones(inputs.shape)*128
    result = net(inputs)
    torch.onnx.export(net, inputs, "refine_resnet18.onnx", verbose=True)
    # parse model
    if type(inputs) == tuple:
        traced_model = torch.jit.trace(*inputs)(net)
    else:
        traced_model = torch.jit.trace(net,inputs)
    fwd = traced_model._get_method('forward')
    params = list(map(lambda x: x.detach().numpy(), fwd.params()))
    with torch.onnx.set_training(net, False):
        trace, _ = torch.jit.get_trace_graph(net, args=(inputs,))
    torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX)
    graph = trace.graph()
    parser = PytorchParser(pluginOp)
    #generate priors and save    
    priors = getPriorLayer(len(Vehicle['STEPS']), Vehicle['FEATURE_MAPS'], Vehicle['SIZE'], Vehicle['STEPS'], Vehicle['MIN_SIZES'], Vehicle['MAX_SIZES'], Vehicle['ASPECT_RATIOS'], Vehicle['USE_MAX_SIZE'], Vehicle['CLIP'])
    inputs = (inputs,priors)
    input_names = ("data","prior")
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as trt_network:
        if type(inputs) == tuple:
            names = [str(i) for i in range(len(inputs))]
        else:
            names = "0"
        parser.RefinedetParse(graph, params, trt_network, inputs, names, Vehicle)
        builder.max_batch_size = batch_size
        builder.max_workspace_size = 1 << 30
        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
         
        engine = builder.build_cuda_engine(trt_network)
        save_engine(engine, "refinedet.engine")
        trt_outputs = inference(trt_network, engine, inputs)
        trt_outputs = trt_outputs[0].host
        draw(img,trt_outputs,int(trt_outputs.shape[0]/7))
