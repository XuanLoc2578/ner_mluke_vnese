FROM python:3.7
#LABEL maintainer="NGUYEN XUAN LOC"

# Environment
ENV LANG=en_US.utf8
ENV PATH=$PATH:/usr/local/cuda-10.0/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
ENV LANG C.UTF-8

# Variables
#ENV NER_MLUKE_DATA=/home/nxloc/Desktop/dataset/ner2016_v2
#ENV NER_MLUKE_MODEL=/home/nxloc/PycharmProjects_clone/ner_mluke_vnese/ner_mluke
#ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Install basic packages and miscellaneous dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    wget \
    curl \
    bzip2 \
    vim \
    ffmpeg \
    unzip \
    alien \
    libaio1\
    libsm6 libxext6 libxrender-dev\
    git \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
RUN apt update -y

# Create env
RUN conda clean --all --yes
RUN conda create -n vnpt python=3.7
Run echo "source activate vnpt" > ~/.bashrc
ENV PATH /opt/conda/envs/vnpt/bin:$PATH

#run!!!
COPY setup.py /setup.py
Run /bin/bash -c "source activate vnpt"

COPY requirements.txt /requirements.txt
Run /bin/bash -c "source activate vnpt && \
    pip install -r requirements.txt"

Run /bin/bash -c "source activate vnpt && \
    python setup.py build_ext && \
    pip3 install ."

ADD mounts /mounts
ADD ner_mluke /ner_mluke

RUN /bin/bash -c "source activate vnpt && \
	python setup.py build_ext --inplace && \
	pip install -e ."
	
WORKDIR /ner_mluke

Run /bin/bash -c "source activate vnpt"

CMD ["sh", "-c", "PYTHONIOENCODING='UTF-8' /bin/bash -c 'source activate vnpt && python ner_mluke/run_train.py'"]

