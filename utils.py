import numpy as np
from stable_baselines.results_plotter import ts2xy, load_results
import matplotlib.pyplot as plt


# def subsample(t, vt, bins):
#     """Given a data such that value vt[i] was observed at time t[i],
#     group it into bins: (bins[j], bins[j+1]) such that values
#     for bin j is equal to average of all vt[k], such that
#     bin[j] <= t[k] < bin[j+1].
#
#     Parameters
#     ----------
#     t: np.array
#         times at which the values are observed
#     vt: np.array
#         values for those times
#     bins: np.array
#         endpoints of the bins.
#         for n bins it shall be of length n + 1.
#
#     Returns
#     -------
#     x: np.array
#         endspoints of all the bins
#     y: np.array
#         average values in all bins"""
#     bin_idx = np.digitize(t, bins) - 1
#     print(bin_idx)
#     v_sums = np.zeros(len(bins), dtype=np.float32)
#     v_cnts = np.zeros(len(bins), dtype=np.float32)
#     print(len(v_sums), len(bin_idx), len(vt))
#     np.add.at(v_sums, bin_idx, vt)
#     np.add.at(v_cnts, bin_idx, 1)
#
#     # ensure graph has no holes
#     zs = np.where(v_cnts == 0)
#     # assert v_cnts[0] > 0
#     for zero_idx in zs:
#         v_sums[zero_idx] = v_sums[zero_idx - 1]
#         v_cnts[zero_idx] = v_cnts[zero_idx - 1]
#
#     return bins, v_sums / v_cnts

def subsample(t, vt, bins):
    """
    Given a data such that value vt[i] was observed at time t[i],
    group it into bins: (bins[j], bins[j+1]) such that values
    for bin j is equal to average of all vt[k], such that
    bin[j] <= t[k] < bin[j+1].
    Parameters
    ----------
    t: np.array
        times at which the values are observed
    vt: np.array
        values for those times
    bins: np.array
        endpoints of the bins.
        for n bins it shall be of length n + 1.
    Returns
    -------
    x: np.array
        endspoints of all the bins
    y: np.array
        average values in all bins
    """
    bin_idx = np.digitize(t, bins) - 1

    v_sums = np.zeros(len(bins), dtype=np.float32)
    v_cnts = np.zeros(len(bins), dtype=np.float32)

    # np.add.at(v_sums, bin_idx, vt)
    # fix np.add.at(v_sums, bin_idx, vt)
    for index, t_iter in enumerate(t):
        binIndex = np.digitize(t_iter, bins) - 1
        np.add.at(v_sums, [binIndex], vt[index])
    np.add.at(v_cnts, bin_idx, 1)

    # ensure graph has no holes
    zs = np.where(v_cnts == 0)[0]

    # v_sums = np.delete(v_sums, zs)
    # v_cnts = np.delete(v_cnts, zs)
    # bins = np.delete(bins, zs)

    # assert v_cnts[0] > 0
    for zero_idx in zs:
        v_sums[zero_idx] = v_sums[zero_idx - 1]
        v_cnts[zero_idx] = v_cnts[zero_idx - 1]
    zs = np.where(v_cnts == 0)[0]
    print('If zs is not Null,v_cnts have zeros', zs)
    return bins[1:], (v_sums / (v_cnts + int(1e-7)))[1:]


def movingAverage(values, window):
    # smooth valus by doing a moving Average
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_result(log_dir, title='Learning Curve'):
    # print(np.cumsum(load_results(log_dir).1.values))
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    # print(x, y)
    # y = movingAverage(y, window=100)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + ' Smoothed')
    plt.show()


def tsplot_result(log_dirs_dict, num_timesteps, title='Learning Curve'):
    # print('load_results', load_results(log_dir))
    import seaborn as sns
    import pandas as pd
    datas = []
    for key in log_dirs_dict:
        log_dirs = log_dirs_dict[key]
        for index, dir in enumerate(log_dirs):
            init_data = load_results(dir)
            init_data = init_data[init_data.l.cumsum() <= num_timesteps]
            x, y = ts2xy(init_data, 'timesteps')
            y = movingAverage(y, window=200)
            x = x[len(x) - len(y):]
            # x = x[:len(y)]
            print('y', y)
            x, y = subsample(t=x, vt=y, bins=np.linspace(0, num_timesteps, int(1000) + 1))
            x = np.append(x, np.array([0]))
            y = np.append(y, np.array([0]))
            print('y after subsample', y)

            # y = movingAverage(y, window=10)
            # # x = x[len(x) - len(y):]
            # x = x[:len(y)]
            data = pd.DataFrame({'Timesteps': x,  'Reward': y, 'subject': np.repeat(index, len(x)), 'Algorithm': np.repeat(key, len(x))})

            datas.append(data)

    data_df = pd.concat(datas, ignore_index=True)

    print('data', data_df)
    sns.tsplot(data=data_df, time='Timesteps', value='Reward', unit='subject', condition='Algorithm')


def read_data(fpath, window=60):
    with open(fpath, 'r') as file:
        raw_data = list(map(float, file.read().split()))
    # end = len(raw_data) // span
    # return [sum(raw_data[span*i: span*(i+1)]) / span for i in range(end)]
    return movingAverage(raw_data,window)

def tsplot_result2(log_dirs_dict, num_timesteps, title='Learning Curve'):
    # print('load_results', load_results(log_dir))
    import seaborn as sns
    import pandas as pd
    datas = []
    for key in log_dirs_dict:
        log_dirs = log_dirs_dict[key]
        for index, dir in enumerate(log_dirs):
            y = read_data(dir)
            y = y[0:num_timesteps]
            x = range(len(y))
            # x = x[:len(y)]
            print('y', y)
            # x, y = subsample(t=x, vt=y, bins=np.linspace(0, num_timesteps, int(1000) + 1))
            # x = np.append(x, np.array([0]))
            # y = np.append(y, np.array([0]))
            # print('y after subsample', y)

            # y = movingAverage(y, window=10)
            # # x = x[len(x) - len(y):]
            # x = x[:len(y)]
            data = pd.DataFrame({'Timesteps': x,  'Reward': y, 'subject': np.repeat(index, len(x)), 'Algorithm': np.repeat(key, len(x))})

            datas.append(data)

    data_df = pd.concat(datas, ignore_index=True)

    print('data', data_df)
    sns.tsplot(data=data_df, time='Timesteps', value='Reward', unit='subject', condition='Algorithm')


if __name__ == '__main__':
    # plot_result(log_dir='test/')
    # import seaborn as sns
    # gammas = sns.load_dataset("gammas")
    # #
    # print('gammas', gammas)
    # ax = sns.tsplot(time="timepoint", value="BOLD signal", unit="subject", condition="ROI", data=gammas)

    # tsplot_result(log_dirs=['test/CartPole/', 'test/dqn_breakout/'])
    # log_dirs = {'A2C_Attention': ['attention_exp1/A2C_Attention_Qbert1'], 'A2C': ['attention_exp1/A2C_Qbert', 'attention_exp1/A2C_Qbert1']}
    # log_dirs = {'A2C_Attention4_1': ['attention_exp/A2C_Attention4_1_Seaquest'],'A2C_Attention4': ['attention_exp/A2C_Attention4_Seaquest'],'A2C_Attention3': ['attention_exp/A2C_Attention3_Seaquest'],'A2C_Attention': ['attention_exp/A2C_Attention_Seaquest'], 'A2C': ['attention_exp/A2C_Seaquest']}
    # log_dirs = {'A2C_Attention4_1': ['attention_exp/A2C_Attention4_1_Seaquest'],'A2C_Attention4': ['attention_exp/A2C_Attention4_Seaquest'],'A2C_Attention3': ['attention_exp/A2C_Attention3_Seaquest'],'A2C_Attention': ['attention_exp/A2C_Attention_Seaquest'], 'A2C': ['attention_exp_new/A2C_Seaquest']}
    
    log_dirs = {'A2C + TP':['LiveStream_0105/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt','LiveStream_0105night/A2CCust6_m10_zhongwang_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt','LiveStream_0104/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn32_action03/reward_during_training.txt'],'A2C': ['LiveStream_0111/A2CCust6_m10_zhongwang_predicttp_pfn32_action03/reward_during_training.txt','LiveStream_0111/A2CCust6_m10_zhongwang_predicttp_pfn32_action03_3/reward_during_training.txt','LiveStream_0111/A2CCust6_m10_zhongwang_predicttp_pfn32_action03_2/reward_during_training.txt' ]}#
    tsplot_result2(log_dirs_dict=log_dirs, num_timesteps=int(270))

    # log_dirs = {'A2C_Attention3': ['attention_exp/A2C_Attention3_Seaquest'],'A2C_Attention': ['attention_exp/A2C_Attention_Seaquest'], 'A2C': ['attention_exp/A2C_Seaquest']}
    # tsplot_result(log_dirs_dict=log_dirs, num_timesteps=int(1e7))

    # print(np.cumsum(load_results('test/').l))P
    # from stable_baselines.results_plotter import plot_results
    # plot_results(['test/', 'test/CartPole/'], int(10e6), 'timesteps', 'Learning Curve')

    # import seaborn as sns
    # import pandas as pd
    # x = np.linspace(0, 15, 31)
    # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    # print(data.shape)
    # print(pd.DataFrame(data).head())  # 每一行数据是一个变量，31列是代表有31天或31种情况下的观测值。
    # # 创建数
    #
    # sns.tsplot(data=data,
    #            err_style="ci_band",   # 误差数据风格，可选：ci_band, ci_bars, boot_traces, boot_kde, unit_traces, unit_points
    #            interpolate=True,      # 是否连线
    #            # ci=[40, 70, 90],       # 设置误差 置信区间
    #            color='g'            # 设置颜色
    #            )
    plt.show()
    print('end')
