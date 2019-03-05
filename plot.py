import matplotlib.pyplot as plt
import numpy as np

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def read_data(fpath, window=60):
    with open(fpath, 'r') as file:
        raw_data = list(map(float, file.read().split()))
    # end = len(raw_data) // span
    # return [sum(raw_data[span*i: span*(i+1)]) / span for i in range(end)]
    return moving_average(raw_data,window)


# def plot(data):
#     file_path = 'LiveStream/a2c_Custom_live_oneEnv/reward_during_training.txt'
#     init = read_data(file_path, 5)
#     x = list(range(len(data)))
#     init = init[:len(data)]
#
#     plt.plot(x, init, '-r')
#     plt.plot(x, data, '-g')
#     # plt.ylim(-1000, 100)
#     plt.show()

def plot(buffer):
    for i, data in enumerate(buffer):
        # data = data[100:]
        plt.plot(range(len(data)), data, label=i+1)



while True:
    file_path = './LiveStream_1229/A2CCust3_zhongwang/reward_during_training.txt'
    final_0 = read_data(file_path)

    file_path = './LiveStream_1229/A2CCust6_deletem8_zhongwang_diff_delay_frame_predicttp/reward_during_training.txt'
    final_1 = read_data(file_path)

    file_path = './LiveStream_1229/A2CCust3_deletem8_zhongwang_diff_delay_frame/reward_during_training.txt'
    final_2 = read_data(file_path)

    file_path = './LiveStream_1229/A2CCust6_deletem8_zhongwang_predicttp/reward_during_training.txt'
    final_3 = read_data(file_path)

    file_path = './LiveStream_1229/A2CCust3_deletem8_zhongwang/reward_during_training.txt'
    final_4 = read_data(file_path)

    file_path = './LiveStream_1229/A2CCust3_deletem8_zhongwang_diff_delay/reward_during_training.txt'
    final_5 = read_data(file_path)

    file_path = './LiveStream_1231/A2CCust6_deletem8_zhongwang_predicttp_tpCoef0.1/reward_during_training.txt'
    final_6 = read_data(file_path)

    file_path = './LiveStream_1231/A2CCust6_m8_zhongwang_predicttp_tpCoef0.1/reward_during_training.txt'
    final_7 = read_data(file_path)

    file_path = './LiveStream_1231/A2CCust6_m8_zhongwang_predicttp_tpCoef0.1_clipR/reward_during_training.txt'
    final_8 = read_data(file_path)

    file_path = './LiveStream_0101/A2CCust6_m8_zhongwang_predicttp_tpCoef0.1_delaym7/reward_during_training.txt'
    final_9 = read_data(file_path)

    file_path = './LiveStream_0101/A2CCust6_m10_zhongwang_predicttp_tpCoef0.1_delaym7/reward_during_training.txt'
    final_10 = read_data(file_path)

    file_path = './test2/reward_during_training.txt'
    final_11 = read_data(file_path)

    file_path = './test/reward_during_training.txt'
    final_12 = read_data(file_path)

    file_path = './LiveStream_0101/A2CMLPLSTM_m10_zhongwang_predicttp_tpCoef0.1/reward_during_training.txt'
    final_13 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef0.1_actionSpace18/reward_during_training.txt'
    final_14 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef1/reward_during_training.txt'
    final_15 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef1_actionSpace18/reward_during_training.txt'
    final_16 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef5/reward_during_training.txt'
    final_17 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef5_actionSame/reward_during_training.txt'
    final_18 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef5_action0_3/reward_during_training.txt'
    final_19 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef5_action1_4/reward_during_training.txt'
    final_20 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef5_action0.5_3.5/reward_during_training.txt'
    final_21 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef1_actionsame/reward_during_training.txt'
    final_22 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef1_action0_5/reward_during_training.txt'
    final_23 = read_data(file_path)

    file_path = './LiveStream_0102/A2CCust6_m10_zhongwang_predicttp_tpCoef1_action0_2/reward_during_training.txt'
    final_24 = read_data(file_path)

    buffer = [final_7,final_8,final_9,final_10,final_12,final_11,final_13]

    buffer = [final_7, final_8, final_12, final_11, final_13,final_14,final_15,final_16,final_17,final_18,final_19,final_20]

    buffer = [final_15, final_16, final_17, final_18,final_19, final_20,final_21,final_22,final_23,final_24]

    # best list
    # buffer = [final_7,final_15,final_22,final_18,final_19,final_17]

    #same 18 action_0_3
    file_path = './LiveStream_0103/A2CCust6_m10_zhongwang_predicttp_tpCoef5/reward_during_training.txt'
    final_25 = read_data(file_path)

    # action_0_3
    file_path = './LiveStream_0103/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn32/reward_during_training.txt'
    final_26 = read_data(file_path)

    # action_0_3
    file_path = './LiveStream_0103/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn64/reward_during_training.txt'
    final_27 = read_data(file_path)

    # action_0_4
    file_path = './LiveStream_0103/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn64_action04/reward_during_training.txt'
    final_28 = read_data(file_path)

    # action_0_4 same 18
    file_path = './LiveStream_0103/A2CCust6_m10_zhongwang_predicttp_tpCoef5_action04/reward_during_training.txt'
    final_29 = read_data(file_path)

    # action_0_4 same 29
    file_path = './LiveStream_0104/A2CCust6_m10_zhongwang_predicttp_tpCoef5_action04/reward_during_training.txt'
    final_30 = read_data(file_path)

    #same 18
    # buffer = [final_18, final_25,final_26,final_27,final_28,final_29,final_30]

    # action_0_3 same 27
    file_path = './LiveStream_0104/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt'
    final_31 = read_data(file_path)

    # buffer = [final_18, final_25, final_26, final_27, final_28, final_29, final_30, final_31]

    # buffer = [final_26, final_27,final_31]

    # add test online
    file_path = './LiveStream_0104/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn32_action03/reward_during_training.txt'
    finalt_1 = read_data(file_path)

    file_path = './LiveStream_0104/A2CCust6_m10_qiangwang_predicttp_tpCoef5_pfn32_action03/reward_during_training.txt'
    finalt_2 = read_data(file_path)

    file_path = './LiveStream_0104/A2CCust6_m10_ruowang_predicttp_tpCoef5_pfn32_action03/reward_during_training.txt'
    finalt_3 = read_data(file_path)

    # buffer = [final_26, final_27, final_31, finalt_1]

    file_path = './LiveStream_0105/A2CCust6_m10_qiangwang_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt'
    finalt_4 = read_data(file_path)

    file_path = './LiveStream_0105/A2CCust6_m10_zhongwang_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt'
    finalt_5 = read_data(file_path)

    file_path = './LiveStream_0105/A2CCust6_m10_ruowang_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt'
    finalt_6 = read_data(file_path)

    file_path = './LiveStream_0105/A2CCust6_m10_netMeter_predicttp_tpCoef5_pfn64_action03/reward_during_training.txt'
    finalt_7 = read_data(file_path)

    buffer = [finalt_2,finalt_1, finalt_3,finalt_4,finalt_5,finalt_6,finalt_7]

    file_path = './LiveStream_0105night/A2CCust6_m10_zhongwang_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_8 = read_data(file_path)

    file_path = './LiveStream_0105night/A2CCust6_m10_qiangwang_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_9 = read_data(file_path)

    file_path = './LiveStream_0105night/A2CCust6_m10_ruowang_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_10 = read_data(file_path)

    file_path = './LiveStream_0105night/A2CCust6_m10_meter_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_11 = read_data(file_path)

    file_path = './LiveStream_0105night/A2CCust6_m10_meterNoweak_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_12 = read_data(file_path)




    file_path = './LiveStream_0106/A2CCust6_m10_meter_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_13 = read_data(file_path)

    file_path = './LiveStream_0106/A2CCust6_m10_meterNoweak_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_14 = read_data(file_path)

    file_path = './LiveStream_0106/A2CCust6_m10_meter_MoreVideo_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_15 = read_data(file_path)

    file_path = './LiveStream_0106/A2CCust6_m10_meterNoweak_MoreVideo_predicttp_tpCoef2_pfn32_action03/reward_during_training.txt'
    finalt_16 = read_data(file_path)

    buffer = [finalt_6, finalt_8, finalt_9, finalt_10, finalt_11, finalt_12, finalt_13, finalt_14,finalt_15,finalt_16]

    buffer = [final_0,final_1,final_2,final_3,final_4]
    plt.ion()
    plt.clf()
    plot(buffer)
    plt.legend()
    plt.show()
    plt.pause(60)
    # plt.ioff()
