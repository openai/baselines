#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line


docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/root/code/stable-baselines,type=bind araffin/stable-baselines-cpu\
  bash -c "cd /root/code/stable-baselines/ && $cmd_line"
