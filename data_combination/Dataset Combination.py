import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import wfdb
import time

start_time = time.time()

# type = []
rootdir = 'mit-bih-arrhythmia-database-1.0.0'
files = os.listdir(rootdir)
name_list = []
MLII = []
type = {}

for file in files:
    if file[0:3] in name_list:
        continue
    else:
        name_list.append(file[0:3])

for name in name_list:
    if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        continue
    record = wfdb.rdrecord(rootdir + '/' + name)

    if 'MLII' in record.sig_name:
        MLII.append(name)
        annotation = wfdb.rdann(rootdir+'/'+name, 'atr')
    for symbol in annotation.symbol:
        if symbol in list(type.keys()):
            type[symbol] += 1
        else:
            type[symbol] = 1
    print('symbol_name', type)

f = 360
segment_len = 10
label_count = 0
count = 0

segmented_data = []
segmented_label = []
print('begin')



for person in MLII:
    k = 0
    whole_signal = wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()
    # lead_index = 0
    # sig_len = wfdb.rdrecord(rootdir + '/' + person).sig_len
    # fs = wfdb.rdrecord(rootdir + '/' + person).fs
    # time = np.arange(sig_len) / fs
    #
    # plt.figure(figsize=(12, 6))  # 设置图形窗口的尺寸
    # plt.plot(time, whole_signal[:, lead_index], label='MLII Signal')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('MLII Signal')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    while (k + 1) * f * segment_len <= len(whole_signal[0]):
        # count += 1
        # k += 1
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len) #3600个点的值
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len)
        lead_index = record.sig_name.index('MLII')

        signal = record.p_signal.transpose()
        # signal_plot = record.p_signal
        # lead_index_before = 0
        # 重采样前的图形

        # sig_len = record.sig_len
        # fs = record.fs
        # time_before = np.arange(sig_len) / fs
        # plt.figure(figsize=(100, 4))  # 设置图形窗口的尺寸
        # plt.plot(time_before, signal_plot[:, lead_index], label='MLII Signal')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('MLII Signal')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


        label = []
        symbols = annotation.symbol
        re_signal = scipy.signal.resample(signal[lead_index], 3600)
        re_signal_3 = np.round(re_signal, 3)
        segmented_data.append(re_signal_3)

        # 画出重采样后的图形
        # time = np.arange(len(re_signal)) / 360.0
        #
        # plt.figure(figsize=(100, 4))
        # plt.plot(time, re_signal, label='Resampled Signal')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Resampled Signal')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        if len(symbols) == 0:
            segmented_label.append('Q')
        elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
                segmented_label.append('N')
        else:
            # for i in symbols:
            #     if i != 'N':
            #         label.append(i)
            # segmented_label.append(label[0])
            # non_n_symbols = [s for s in symbols if s != 'N']
            #
            # if len(non_n_symbols) == 0:
            #     segmented_label.append('N')  # 如果没有非'N'的标签，标记为'N'
            # else:
            #     # 统计出现次数最多的非'N'标签
            #     most_common_symbol = max(set(non_n_symbols), key=non_n_symbols.count)
            #     segmented_label.append(most_common_symbol)
            non_n_symbols = [s for s in symbols if s != 'N']

            if '+' in symbols:
                segmented_label.append('+')  # 如果包括'+'，直接标记为'+'
            elif len(non_n_symbols) == 0:
                segmented_label.append('N')  # 如果没有非'N'的标签，标记为'N'
            else:
                # 统计出现次数最多的非'N'标签
                most_common_symbol = max(set(non_n_symbols), key=non_n_symbols.count)
                segmented_label.append(most_common_symbol)
        k += 1




# time = np.arange(len(re_signal)) / 360.0

# 绘制重采样后的信号
# plt.figure(figsize=(10, 4))
# plt.plot(time, re_signal, label='Resampled Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Resampled Signal')
# plt.legend()
# plt.grid(True)
# plt.show()

#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"程序运行时间：{execution_time} 秒")
