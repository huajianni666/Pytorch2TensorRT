import torch
import torchvision

# 获取模型实例
model = torchvision.models.resnet18()

# 生成一个样本供网络前向传播 forward()
example = torch.rand(1, 3, 224, 224)

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output[0, :5])
traced_script_module.save("../scriptmodule/resnet-18-model.pt")
