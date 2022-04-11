FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION 10.1
ENV PYTHON_VERSION 3.7.0

WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev

RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall

RUN ln -s /usr/local/bin/python3.7 /bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /bin/pip3
# install pip
ENV KERAS_VERSION 2.3.1
ENV TF_VERSION 2.3.0

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==${TF_VERSION}
RUN pip3 install keras==${KERAS_VERSION}
RUN pip3 install scikit-learn matplotlib tqdm pandas scikit-image opencv-python psutil
RUN pip3 install tensorflow-model-optimization
CMD [ "/bin/bash" ]
