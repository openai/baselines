#!/usr/bin/env bash

DOCKER_CMD="docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/code/stable-baselines,type=bind"
BASH_CMD="cd /root/code/stable-baselines/"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <test glob>"
  exit 1
fi

if [[ ${DOCKER_IMAGE} = "" ]]; then
  echo "Need DOCKER_IMAGE environment variable to be set."
  exit 1
fi

TEST_GLOB=$1

set -e  # exit immediately on any error

# For pull requests from fork, Codacy token is not available, leading to build failure
if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
  ${DOCKER_CMD} ${DOCKER_IMAGE} \
      bash -c "${BASH_CMD} && \
               pytest --cov-config .coveragerc --cov-report term --cov=. -v tests/test_${TEST_GLOB}"
else
  if [[ ${CODACY_PROJECT_TOKEN} = "" ]]; then
    echo "Need CODACY_PROJECT_TOKEN environment variable to be set."
    exit 1
  fi

  ${DOCKER_CMD} --env CODACY_PROJECT_TOKEN=${CODACY_PROJECT_TOKEN} ${DOCKER_IMAGE} \
      bash -c "${BASH_CMD} && \
                pytest --cov-config .coveragerc --cov-report term --cov-report xml --cov=. -v tests/test_${TEST_GLOB} && \
                /root/code/codacy-coverage-reporter report -l python -r coverage.xml --partial"
fi
