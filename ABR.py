# import pyparams
import numpy as np
import time
from ToyEnv4ABR import ToyEnv
from stable_baselines import A2C
from stable_baselines.common.policies import LstmCust6Policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds

def make_env(rank, log_dir='test', seed=0):
    def _init():
        env = ToyEnv(train=True, log_dir=log_dir)
        # env.seed(seed + rank)
        return env

    set_global_seeds(seed)

    return _init


num_cpu = 2
past_frame_num = 32
BIT_RATE = [500,1200]

class Algorithm:
     def __init__(self,model,Env):
     # fill your init vars
     #     env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
     #     # self.model = A2C.load('/home/team/出来挨打/submit/model'
     #     #                       , env=env, policy=LstmCustPolicy)
     #     self.model = A2C.load('/home/www/RL/LiveStreamingDemo/final/submit/model.pkl',
     #                           env=env, policy=LstmCust6Policy)

         # state_ = [0] * 38
         # self.model.action_probability([state_] * num_cpu)
         # self.model.predict([state_] * num_cpu)
         self.model = model
         self.env = Env
         state_ = [0] * (past_frame_num*2+6)
         self.model.action_probability([state_] * num_cpu)

     # Intial 
     def Initial(self):
         self.buffer_size = 0
         self.last_bit_rate = 0
         self.switch_num = 0
         self.last_target_buffer = 0
         self.state_history = [[0] * 2 for i in range(past_frame_num)]
         self.hidden_states = None

     #Define your al
     def run(self, S_time_interval, S_send_data_size, S_frame_time_len, S_frame_type, S_buffer_size, S_end_delay, rebuf_time, cdn_has_frame,cdn_flag, buffer_flag):
         # record your params
         time_s = time.time()
         assert sum(S_time_interval) > 0.5
         call_time = 0
         for i in range(len(S_time_interval)-1, -1, -1):
             call_time += S_time_interval[i]
             if call_time > 0.5:
                 break

         sum_video_size = sum(S_send_data_size[i:])
         sum_download_time = call_time
         throughput = sum_video_size / (sum_download_time + 1e-7) / 1e6
         download_time = sum_download_time
         buffer_size = S_buffer_size[-1]
         last_bit_rate = BIT_RATE[self.last_bit_rate] / BIT_RATE[-1]
         last_target_buffer = self.last_target_buffer
         rebuf = rebuf_time
         switch_num = self.switch_num
         # rebuf = last_rebuf
         n_chunk = len(S_time_interval) - i + 1
         end_delay = sum(S_end_delay[i:]) / n_chunk
         self.state_history.append([throughput, download_time])
         self.state_history.pop(0)
         instant_state = [buffer_size, last_bit_rate, rebuf, switch_num, end_delay, last_target_buffer]
         state_ = np.reshape(list(zip(*self.state_history)), -1).tolist() + instant_state


         action_probability = self.model.action_probability([state_] * num_cpu, self.hidden_states)
         action = np.argmax(action_probability, axis=1)[0]
         # _, self.hidden_states = self.model.predict([state_] * num_cpu)
         # your decision
         bit_rate, target_buffer = action // 4, action % 4

         self.switch_num = abs(bit_rate - self.last_bit_rate)
         self.last_bit_rate = bit_rate
         self.last_target_buffer = target_buffer

         # print(bit_rate,target_buffer)

         return bit_rate, target_buffer


