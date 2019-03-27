FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer = "Qiniu ATLab<ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libxine2-dev \
    libv4l-dev \
    libgstreamer1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    wget \
    curl \
    vim \
    zip \
    unzip \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    git \
    ca-certificates \
    libprotobuf-dev \
    ffmpeg \
    protobuf-compiler && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install python dependencies
RUN wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python3 get-pip.py && \
    pip3 install --index-url https://mirrors.aliyun.com/pypi/simple/ numpy


ENV TensorRT_FILE=TensorRT-5.0.2.6.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.3.tar.gz

RUN wget http://pbsdv028w.bkt.clouddn.com/softwares/nvidia/$TensorRT_FILE -O /tmp/$TensorRT_FILE && \
    tar xzf /tmp/$TensorRT_FILE -C /tmp && cd /tmp/TensorRT-5.0.2.6 && \
    cp include/Nv* /usr/local/include && cp -P lib/libnv* /usr/local/lib && ldconfig && \
    pip3 install python/tensorrt-5.0.2.6-py2.py3-none-any.whl
#   rm -rf /tmp/TensorRT-5.0.2.6*

# opencv 3
RUN export OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.4.1 && cd /tmp && \
     wget http://pbv7wun2s.bkt.clouddn.com/opencv-3.4.1.tar && tar -xvf opencv-3.4.1.tar && mv opencv-3.4.1 opencv && \
     mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
     cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/ \
     -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=1 -D WITH_NVCUVID=on -D CUDA_GENERATION=Auto .. && \
     make -j"$(nproc)" && make install && ldconfig && \
     rm -rf /tmp/*

# pytorch
RUN pip3 install http://p8jmuamj4.bkt.clouddn.com/torch-0.4.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install --index-url https://mirrors.aliyun.com/pypi/simple/ torchvision pycuda pillow opencv-python


RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "tron"

