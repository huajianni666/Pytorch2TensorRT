# Atlab SDK TEAM Inference LIB
### 运行环境
参考Dockerfile

### 安装
```Shell
git clone --recursive https://github.com/huajianni666/Pytorch2TensorRT.git && cd Pytorch2TensorRT
wget http://pq16f3soz.bkt.clouddn.com/Pytorch2Trt3rdparty.zip
unzip Pytorch2Trt3rdparty.zip && rm Pytorch2Trt3rdparty.zip
```
### 编绎
```Shell
cd python/plugin && mkdir build && cd build && cmake ..
make
cd ../../
```
### 运行
```Shell
python refinedet.py

```
