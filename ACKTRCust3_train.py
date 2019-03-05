import os
import time
from ToyEnv4Cust3 import ToyEnv
from stable_baselines import ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import LstmCust3Policy,MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
import numpy as np
# from simple_online_test4Cust3 import test



def make_env(rank, log_dir,seed=0):
    def _init():
        env = ToyEnv(train=True, log_dir=log_dir,)
        env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)

    return _init


best_mean_reward, min_best_reward, n_steps = [-np.inf]*10, -np.inf, 0


def callback(_locals, _globals):
    global n_steps, best_mean_reward,min_best_reward
    step = n_steps + 1
    if step > 10000:
        fpath = '{}reward_during_training.txt'.format(log_dir)
        with open(fpath, 'r') as file:
            raw_data = list(map(float, file.read().split()))
        data = raw_data[-200:]
        mean_reward = np.mean(data)
        if mean_reward > min_best_reward:
            min_best_reward = mean_reward
            index = np.argmin(best_mean_reward)
            best_mean_reward[index] = mean_reward
            print('best_mean_reward', best_mean_reward)
            _locals['self'].save(log_dir + 'best_model_{}.pkl'.format(str(index)))
    if step % 500 == 0:
        _locals['self'].save(log_dir + 'model.pkl')
    n_steps += 1
    return False

# def callback(_locals, _globals):
#     global n_steps, best_mean_reward,min_best_reward
#     step = n_steps + 1
#     if step > 1.9e5 and step % 100 == 0:
#         mean_reward = test('aaa',_locals['self'],env)
#         index = np.argmin(best_mean_reward)
#         if mean_reward > best_mean_reward[index]:
#             best_mean_reward[index] = mean_reward
#             print('best_mean_reward', best_mean_reward)
#             _locals['self'].save(log_dir + 'best_model_{}.pkl'.format(str(mean_reward)))
#     n_steps += 1
#     return False


# log_dir = 'LiveStream_1229/ACKTRCust3_deletem8_zhongwang_diff_delay/'
log_dir = 'ACKTRtest/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tstart = time.time()
num_cpu = 2

env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

model = ACKTR(env=env,
            policy=LstmCust3Policy,
            verbose=1,
            )

model.learn(total_timesteps=int(5e6), callback=callback)
model.save(log_dir + "last_model")

print('Time taken: {:.2f}'.format(time.time() - tstart))
