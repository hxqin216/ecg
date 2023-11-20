import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import wfdb
import time
import pandas as pd

start_time = time.time()

# type = []
# rootdir = 'mit-bih-st-change-database-1.0.0'
rootdir = 'sudden-cardiac-death-holter-database-1.0.0'
# rootdir = 'european-st-t-database-1.0.0'
# rootdir = 'mit-bih-arrhythmia-database-1.0.0'


files = os.listdir(rootdir)
name_list = []
ECG = []
type = {}

for file in files:
    if file[0:2] in name_list:
        continue
    else:
        name_list.append(file[0:2])

for name in name_list:
    if name[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] or name in ['33', '37', '38', '39', '40', '42', '43', '44', '47', '48', '50' ]:
        continue
    record = wfdb.rdrecord(rootdir + '/' + name)

    if 'ECG' in record.sig_name:
        ECG.append(name)
    annotation = wfdb.rdann(rootdir+'/'+name, 'atr')
    for symbol in annotation.symbol:
        if symbol in list(type.keys()):
            type[symbol] += 1
        else:
            type[symbol] = 1
print('symbol_name', type)

f = 250
segment_len = 10
label_count = 0
count = 0

segmented_data = []
segmented_label = []
print('begin')

for person in ECG:
    k = 0
    whole_signal = wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()

    while (k + 1) * f * segment_len <= len(whole_signal[0]):
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len) #3600个点的值
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segment_len, sampto=(k + 1) * f * segment_len)
        lead_index = record.sig_name.index('ECG')

        signal = record.p_signal.transpose()
        label = []
        symbols = annotation.symbol
        re_signal = scipy.signal.resample(signal[lead_index], 3600)
        re_signal_3 = np.round(re_signal, 3)
        segmented_data.append(re_signal_3)
        if len(symbols) == 0:
            segmented_label.append('Q')
        elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
                segmented_label.append('N')
        else:
            non_n_symbols = [s for s in symbols if s != 'N']
            if len(non_n_symbols) == 0:
                segmented_label.append('N')  # 如果没有非'N'的标签，标记为'N'
            else:
                # 统计出现次数最多的非'N'标签
                most_common_symbol = max(set(non_n_symbols), key=non_n_symbols.count)
                segmented_label.append(most_common_symbol)
        ass = ((k + 1) * f * segment_len) / len(whole_signal[0])
        print(ass)
        k += 1


# save as csv
segmented_data = pd.DataFrame(segmented_data)
segmented_label = pd.DataFrame(segmented_label)
segmented_data.to_csv('X_SU_ECG_2.csv', index=False)
segmented_label.to_csv('Y_SU_ECG_2.csv', index=False)


end_time = time.time()
execution_time = end_time - start_time
print(f"程序运行时间：{execution_time} 秒")

# # save to mongodb
#
# from pymongo import MongoClient
# import csv
#
# def connection():
#     conn = MongoClient("localhost")
#     db = conn.ECGdb
#     set1 = db.ECG
#
#     return set1
#
# def insertToMongoDB(set1):
#     with open('3Y_eu_MLII.csv', 'r', encoding = 'utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         array_y = []
#         for label in reader:
#             for i in label:
#                 array_y.append(label[i])
#
#     with open('3X_eu_MLII.csv', 'r', encoding = 'utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         counts = 0
#         for signal in reader:
#             array_x = []
#             for k in signal:
#                 array_x.append(float(signal[k]))
#             data = {
#                 'fs': 360,
#                 'from': 'ecg_data',
#                 'lead': 'MLII',
#                 'p_signal': array_x,
#                 'label': array_y[counts],
#                 'resample': False
#             }
#             print(counts)
#             print(array_x)
#             print(array_y[counts])
#
#             counts += 1
#             set1.insert_one(data)
#         print('Successfully added ' + str(counts) + ' records.')
#
# def main():
#     set1=connection()
#     insertToMongoDB(set1)
# # 判断是不是调用的main函数。这样以后调用的时候就可以防止不会多次调用 或者函数调用错误
# if __name__=='__main__':
#     main()


