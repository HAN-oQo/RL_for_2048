# RL for 2048
This repository includes Pytorch implementations of various RL algorithms with custom gym of 2048 game.

---

## Algorithms supported

- [x] DQN
- [ ] SAC
- [ ] DDPG
- [ ] PPO

Rest algorithms will be supported soon.

---
## Set up

Install dependencies with Docker. You can also install dependencies using requirements.txt.

To build Docker image, run this command.
```
# format: docker build -t . <image_name>
docker build -t . 2048
```

After building image, use the following command to run the Docker container.
```
docker run -ti --gpus '"device='<gpu number>'"' -v <your working directory>:/app --ipc=host --name <container_name> <image_name> /bin/bash

# or you can run this command after changing docker_run.sh file in proper format
./docker_run.sh <gpu num> <container_name>
```
---
## Train
If you want to train your own agent, run
```
# For DQN agent,
python train_.py --config configs/config_dqn.yaml
```

You can freely change the hyperparameter if you needed.

---
## Test

You can test with the pretrained networks.
It can be downloaded in following links.
Links will be supported soon.

[DQN]()

To render the playing result with the network, run
```
# before run this command, you should put the path to checkpoint in config file.
# For DQN_agent,
python render.py --config configs/config_dqn.yaml

```

---
## Results

The rendered result of playing 2048 with DQN agent.
![DQN_result](./gifs/game_play_dqn.gif)

should add logs / best score 
add release



---
## Acknowledgement
1. https://github.com/georgwiese/2048-rl
2. https://github.com/navjindervirdee/2048-deep-reinforcement-learning
3. https://github.com/rajitbanerjee/2048-pygame
4. https://www.geeksforgeeks.org/2048-game-in-python/
5. https://github.com/activatedgeek/gym-2048/tree/master/gym_2048
