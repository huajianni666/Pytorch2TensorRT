# Atlab SDK TEAM Inference LIB
### 编绎
```Shell
mkdir build & cd build
cmake -DCMAKE_PREFIX_PATH="3rdparty/libtorch" ..
make
python ../modules/classification/resnet18_torchscript.py 
./resnet18_test ../scriptmodule/resnet-18-model.pt
```
### 安装
1. [http://p3gvzhggv.bkt.clouddn.com/Pytorch2Trt3rdparty.zip](#第三方库)
