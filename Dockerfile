FROM python:3.6

RUN apt-get -y update && apt-get -y install ffmpeg udev strace less sudo libopenmpi-dev libglew-dev patchelf libosmesa-dev
RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv


CMD /bin/bash
ARG USER
ARG HOME
ARG UID
RUN groupadd -g "2000" "$USER" \
 && useradd --uid "$UID" -s "/bin/bash" -c "$USER" -g "2000" -d "$HOME" "$USER" \
 && echo "$USER:$USER" | chpasswd \
 && adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER
# Commands below run as the developer user
USER $USER
ARG PWD
WORKDIR $PWD
ENV CODE_DIR $PWD
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/home/vdhiman/.mujoco/mujoco200/bin

# Clean up pycache and pyc files
RUN sudo pip install tensorflow mpi4py opencv-python click cloudpickle

ENTRYPOINT /bin/bash
