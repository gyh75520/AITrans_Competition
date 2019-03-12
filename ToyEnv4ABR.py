from gym import spaces
import json
import numpy as np
import gym


class ToyEnv(gym.Env):
    def __init__(self, log_dir='', train=False, random_seed=0):
        # all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
        # self.all_cooked_time = all_cooked_time
        # self.all_cooked_bw = all_cooked_bw
        self.observation_space = spaces.Box(low=np.repeat(-np.inf, 38), high=np.repeat(np.inf, 38))
        # self.action_space = spaces.Box(low=-3.7,high=3.7,shape=(1,),dtype=np.float32)
        self.action_space = spaces.Discrete(10)
        # self.cnt = 0
        # self.train = train
        # self.log_dir = log_dir
        # self.seed()

    