from gym import spaces
import json
import numpy as np
import LiveStreamingEnv.final_env as env
import LiveStreamingEnv.load_trace as load_trace
import gym
from gym.utils import seeding

BIT_RATE = [500, 1200]

LogFile_Path = "./log/"
# TRAIN_TRACES = './network_trace/'

TRAIN_TRACES = './new_network/zhongwang/'
# TRAIN_TRACES = './new_network/ruowang/'
# TRAIN_TRACES = './new_network/qiangwang/'
# TRAIN_TRACES = './network_meter/'
# TRAIN_TRACES = './network_meter_Noweak/'

video_size_files = ('./video_trace/12-12/video_trace/frame_trace_',
                    #'./video_trace/12-13/video_trace/frame_trace_',
                    #'./video_trace/12-14/video_trace/frame_trace_',
                    './video_trace/12-16/video_trace/frame_trace_',
                    './video_trace/12-21/video_trace/frame_trace_',
                    # './video_trace/12-24/video_trace/frame_trace_', # more
                    )

# video_size_files = ('./new_video/12-16/video_trace/frame_trace_',
#                     './new_video/12-17/video_trace/frame_trace_',
#                     './new_video/12-18/video_trace/frame_trace_',
#                     './new_video/12-19/video_trace/frame_trace_',
#                     './new_video/12-20/video_trace/frame_trace_',
#                     './new_video/12-21/video_trace/frame_trace_',
#                     './new_video/12-22/video_trace/frame_trace_',
#                     './new_video/12-23/video_trace/frame_trace_',
#                     './new_video/12-24/video_trace/frame_trace_', 1800
#                     )

frame_time_len = 0.04
SMOOTH_PENALTY = 0.02
REBUF_PENALTY = 1.5
LANTENCY_PENALTY = 0.005


class ToyEnv(gym.Env):
    def __init__(self, log_dir='', train=False, random_seed=0):
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.observation_space = spaces.Box(low=np.repeat(-np.inf, 70), high=np.repeat(np.inf, 70))
        # self.action_space = spaces.Box(low=-3.7,high=3.7,shape=(1,),dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self.cnt = 0
        self.train = train
        self.log_dir = log_dir
        self.seed()

    def reset(self):
        frame_trace_id = np.random.randint(0, 3)
        # trace_idx = np.random.randint(len(self.all_cooked_time))
        # frame_trace_id = 2
        self.net_env = env.Environment(all_cooked_time=self.all_cooked_time,
                                       all_cooked_bw=self.all_cooked_bw,
                                       logfile_path=LogFile_Path,
                                       VIDEO_SIZE_FILE=video_size_files[frame_trace_id],
                                       Debug=False)
        self.physical_time = 0
        self.time_record = []
        self.bit_rate_record = []
        self.target_buffer_record = []
        self.buffer_record = []
        self.throughput_record = []

        self.done = False
        self.last_bit_rate = 0
        self.reward_in_episode = 0
        self.last_end_delay = 0

        # reset _states
        self.past_frame_num = 32
        self.state_history = [[0] * 2 for i in range(self.past_frame_num)]
        state_, reward, _ = self.forward(0, 1.5)
        return state_

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def forward(self, bit_rate, target_buffer):
        reward = 0

        sum_video_size = sum_download_time = sum_buffer_size = sum_rebuf  = sum_end_delay = 0
        n_chunk = 0
        last_rebuf = 0
        call_time = 0
        switch_num = 0
        last_end_delay = 0
        while True:
            (time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, end_delay, cdn_newest_id, downlaod_id,
             cdn_has_frame, decision_flag, buffer_flag, switch, cdn_flag, end_of_video) = self.net_env.get_video_frame(bit_rate, target_buffer)

            call_time += time_interval
            switch_num += switch
            n_chunk += 1
            sum_download_time += time_interval
            sum_video_size += send_data_size
            sum_buffer_size += buffer_size
            sum_rebuf += rebuf
            sum_end_delay += end_delay
            # sum_end_delay += end_delay - last_end_delay
            # last_end_delay = end_delay

            if end_delay>7.5:
                print(rebuf,end_delay)
            if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000 \
                    - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)
            if call_time > 0.5 and not end_of_video:
                reward_frame += -(switch_num) * SMOOTH_PENALTY * (1200 - 500) / 1000
                break
            if end_of_video:
                break
            reward += reward_frame
        reward += reward_frame

        throughput = sum_video_size / (sum_download_time + 1e-7) / 1e6
        download_time = sum_download_time
        last_bit_rate = BIT_RATE[bit_rate] / BIT_RATE[-1]
        last_target_buffer = target_buffer
        rebuf = sum_rebuf
        end_delay = sum_end_delay / n_chunk
        # end_delay = sum_end_delay / n_chunk - self.last_end_delay
        # self.last_end_delay = sum_end_delay / n_chunk
        # end_delay = end_delay / 7
        self.state_history.append([throughput, download_time])
        self.state_history.pop(0)
        instant_state = [buffer_size, last_bit_rate, rebuf, switch_num,end_delay, last_target_buffer]
        state_ = np.reshape(list(zip(*self.state_history)), -1).tolist() + instant_state
        self.reward_in_episode += reward

        reward /= n_chunk
        reward *= 10

        # reward = max(reward, -1.)
        if abs(reward) > 1:
            print('REWARD:',reward,'THROUGHPUT:',throughput)


        return state_, reward, end_of_video

    def step(self, action):

        # if action > 0:
        #     bit_rate = 1
        #     target_buffer = action + 0.3
        # else :
        #     bit_rate = 0
        #     target_buffer = -action + 0.3
        # target_buffer = target_buffer[0]

        bit_rate, target_buffer =  action // 4  , action % 4


        # if action > 8:
        #     bit_rate = 1
        #     target_buffer = (action - 9) / 2
        # else :
        #     bit_rate = 0
        #     target_buffer = action / 2


        # print(bit_rate,target_buffer)
        state_, reward, done = self.forward(bit_rate, target_buffer)
        self.done = done
        self.last_bit_rate = bit_rate

        if self.train and done:
            with open(self.log_dir + 'reward_during_training.txt', 'a') as file:
                file.write('{}\n'.format(self.reward_in_episode))
        # if done:
        #     with open('recorders.txt', 'w') as file:
        #         file.write('{}\n'.format(json.dumps(self.reward_in_episode)))
        #         file.write('{}\n'.format(json.dumps(self.time_record)))
        #         file.write('{}\n'.format(json.dumps(self.bit_rate_record)))
        #         file.write('{}\n'.format(json.dumps(self.target_buffer_record)))
        #         file.write('{}\n'.format(json.dumps(self.buffer_record)))
        #         file.write('{}\n'.format(json.dumps(self.throughput_record)))

        return state_, reward, done, {}

    # def get_time_record(self):
    #     return self.time_record
    #
    # def get_bit_rate_record(self):
    #     return self.bit_rate_record
    #
    # def get_target_buffer_record(self):
    #     return self.target_buffer_record
    #
    # def get_buffer_record(self):
    #     return self.buffer_record
    #
    # def get_throughput_record(self):
    #     return self.throughput_record