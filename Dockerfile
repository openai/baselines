FROM ubuntu:16.04

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake
ENV CODE_DIR /root/code
ENV VENV /root/venv

COPY . $CODE_DIR/baselines
RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    pip install -e baselines && \
    pip install pytest

ENV PATH=$VENV/bin:$PATH
WORKDIR $CODE_DIR/baselines

CMD /bin/bash
