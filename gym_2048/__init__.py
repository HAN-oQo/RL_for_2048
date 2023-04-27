from gymnasium.envs.registration import register
from .env import Base2048Env

register(
    id='Tiny2048-v1',
    entry_point='gym_2048.env:Base2048Env',
    kwargs={
        'width': 2,
        'height': 2,
    },
    max_episode_steps= 2000
)

register(
    id='2048-v1',
    entry_point='gym_2048.env:Base2048Env',
    kwargs={
        'width': 4,
        'height': 4,
    },
    max_episode_steps = 2000
)