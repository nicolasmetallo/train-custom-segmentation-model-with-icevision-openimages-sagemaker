FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV PYTHONUNBUFFERED TRUE

# FASTAI
RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz && apt-get update
RUN pip install numpy \
    matplotlib \
    opencv-python \
    pillow

# TORCHSERVE
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    openjdk-11-jdk \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# TORCHSERVE
RUN git clone https://github.com/pytorch/serve.git
RUN pip install ./serve/

COPY ./torchserve_byoc_with_amazon_sagemaker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./torchserve_byoc_with_amazon_sagemaker/config.properties /home/model-server/config.properties

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]