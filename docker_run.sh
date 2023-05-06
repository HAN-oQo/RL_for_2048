#!/bin/bash
docker run -ti --gpus '"device='$1'"' -v /project/RL_for_2048:/app --ipc=host --name $2 2048 /bin/bash
# ex) docker run -ti --gpus '"device='$1'"' -v <put_your_working_directory>:/app --ipc=host --name $2 <put_your_docker_image_name> /bin/bash