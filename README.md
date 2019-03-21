# Atlab SDK TEAM Inference LIB
### 编绎
```Shell
mkdir build & cd build
cmake -DCMAKE_PREFIX_PATH="3rdparty/libtorch" ..
make
python ../modules/classification/resnet18_torchscript.py 
./resnet18_test ../scriptmodule/resnet-18-model.pt
```
