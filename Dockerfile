FROM python:3.6-slim

# Security: Be specific on version, so use ffmpeg=<version>
RUN apt-get update && apt-get -y upgrade && apt-get install -y ffmpeg
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/code

# Security: Preserve the ownership of the files and directories, so that the files are owned by the right user and group.
# So use COPY --chown=<user:group> . $CODE_DIR/baselines
COPY . $CODE_DIR/baselines
WORKDIR $CODE_DIR/baselines

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install 'tensorflow < 2' && \
    pip install -e .[test]
    
RUN apt-get clean
CMD ["/bin/bash"]
