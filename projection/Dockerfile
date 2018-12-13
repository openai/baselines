FROM chainer/chainer:v4.5.0-python3
MAINTAINER Yoshimura Naoya <yoshimurae708x@gmail.com>

# Emacsのインストール
RUN apt-get update
RUN apt-get install emacs24-nox -y


# Install Chainer
RUN pip3 install jupyter \
    && jupyter notebook --generate-config
RUN echo 'alias python=python3' >> ~/.bashrc \
    && echo 'alias pip=pip3' >> ~/.bashrc
RUN pip3 install --upgrade pip

# Install Python Module
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt
# RUN pip3 install matplotlib numpy scipy sklearn pandas seaborn librosa h5py



# Finish
RUN mkdir /root/work
WORKDIR /root/work
